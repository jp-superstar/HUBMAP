import copy
import logging
from enum import Enum

import numpy as np
from PIL import Image as PImage
from pyquaternion import Quaternion

from .utils import BaseGLTFStructure
from .materials import Material, Sampler
from .buffers import Vec3Array, Vec4Array, ScalarArray

logger = logging.getLogger(__name__)


class Primitive(BaseGLTFStructure):
    attributes = None
    mode = None
    material = None
    targets = None
    _indices = None

    class Mode(Enum):
        POINTS = 0
        LINES = 1
        LINE_LOOP = 2
        LINE_STRIP = 3
        TRIANGLES = 4
        TRIANGLE_STRIP = 5
        TRIANGLE_FAN = 6

    def __init__(self, prim=None, gltf=None, name=None, indices=None,
                 material=None, image=None, texture=None, mode=None,
                 attributes=None, targets=None, **kwargs):
        self.name = name
        if prim:
            if 'name' in prim:
                self.name = name
            if 'material' in prim:
                self.material = gltf.materials[prim.get('material')]
            if 'indices' in prim:
                self.indices = gltf.accessors[prim.get('indices')]
            if 'mode' in prim:
                self.mode = self.Mode(prim.get('mode'))
            attributes = prim.get('attributes')
            if attributes:
                self.attributes = Attributes(attributes, gltf)
            targets = prim.get('targets')
            if targets:
                self.targets = [Attributes(t, gltf) for t in targets]
        else:
            self.material = material
            self.mode = mode and self.Mode[mode]
            if image:
                self.material = Material(image=image)
            if texture:
                self.material = Material(color_texture=texture)
            if attributes:
                self.attributes = attributes
            elif kwargs:
                self.attributes = Attributes(gltf=gltf, **kwargs)
            else:
                self.attributes = Attributes()
            if targets:
                self.targets = targets
            if indices:
                self.indices = indices

    def __repr__(self):
        return self.name or super(Primitive, self).__repr__()

    def __contains__(self, item):
        return (self._indices == item
                or item in self.attributes
                or (self.targets and any(item in t for t in self.targets)))

    def sort_joints(self):
        self.attributes.sort_joints()

    def split_transparency(self, alpha_mode=Material.AlphaMode.BLEND):
        if self.mode and self.mode != self.Mode.TRIANGLES:
            raise NotImplementedError('Prim splitting only implemented for triangles')

        if not self.material or not self.attributes.texcoords or (
                self.material.alpha_mode == Material.AlphaMode.OPAQUE):
            raise ValueError('Can\'t split prim: no material or texcoords, or material '
                             'is opaque, or material has no color_texture or diffuse_texture.')

        tex = self.material.color_texture or self.material.diffuse_texture
        if tex is None:
            return
        sampler = tex.sampler or Sampler()
        try:
            fp = tex.source.get_fp()
        except ValueError:
            return
        img = PImage.open(fp)
        if len(img.getbands()) != 4:
            return
        alpha = img.getdata(3)
        min_alpha, max_alpha = alpha.getextrema()

        # return if all or none of the image is transparent
        if min_alpha == 255 or max_alpha < 255:
            return

        if self.material.color_uv is not None:
            texcoords = self.attributes.texcoords[self.material.color_uv]
        else:
            texcoords = self.attributes.texcoords[0]

        if self.indices:
            indices_iter = (self.indices.data[pos:pos + 3]
                            for pos in
                            range(0, self.indices.count, 3))
        else:
            indices_iter = ((i, i + 1, i + 2)
                            for i in
                            range(0, texcoords.count, 3))
        t_indices = []
        o_indices = []
        for indices in indices_iter:
            for point in [texcoords.data[i] for i in indices]:
                point = sampler.wrap_point(point)
                x = round((alpha.size[0] - 1) * point[0])
                y = round((alpha.size[1] - 1) * point[1])
                alpha_val = alpha.getpixel((x, y))
                if alpha_val < 255:
                    break
            else:
                o_indices.extend(indices)
                continue
            t_indices.extend(indices)

        # return if there are no opaque vertices
        if not len(o_indices):
            return

        t_material = copy.copy(self.material)
        t_material.alpha_mode = alpha_mode
        # Copy own material in case something else is using it
        self.material = copy.copy(self.material)
        self.material.alpha_mode = Material.AlphaMode.OPAQUE
        self.indices = o_indices


        attributes = Attributes(positions=self.attributes.positions, normals=self.attributes.normals,
                                texcoords=self.attributes.texcoords, tangents=self.attributes.tangents,
                                joints=self.attributes.joints, weights=self.attributes.weights, colors=self.attributes.colors)
        return Primitive(attributes=attributes, indices=t_indices, material=t_material)

    def render(self, gltf):
        primitive = {}
        if self.mode:
            primitive['mode'] = self.mode.value
        if self.material:
            primitive['material'] = gltf.index('materials', self.material)
        if self.indices and self.indices.count:
            primitive['indices'] = gltf.index('accessors', self.indices)
        if self.attributes:
            primitive['attributes'] = self.attributes.render(gltf)
        if self.targets:
            targets = []
            for target in self.targets:
                targets.append(target.render(gltf))
            primitive['targets'] = targets
        return primitive

    def repair(self):
        return
        if self.normals:
            magnitudes = np.sqrt((self.normals.data ** 2).sum(-1))[..., np.newaxis]

            if not np.all(np.isclose(magnitudes, 1)):
                self.normals.data /= magnitudes

            if np.any(magnitudes == 0):
                zero_indices, _ = np.where(magnitudes == 0)
                for index in zero_indices:
                    if self.indices:
                        matching = np.where(self.indices.data == index)[0]
                        if not matching.size:
                            # normal unused anyways
                            self.normals.data[index] = [0, 1, 0]
                            continue
                        index_in_indices = matching[0]
                        mod = index_in_indices % 3
                        a = self.positions.data[self.indices.data[index_in_indices - mod]]
                        b = self.positions.data[self.indices.data[1 + index_in_indices - mod]]
                        c = self.positions.data[self.indices.data[2 + index_in_indices - mod]]
                    else:
                        mod = index % 3
                        a = self.positions.data[index - mod]
                        b = self.positions.data[1 + index - mod]
                        c = self.positions.data[2 + index - mod]
                    dir = np.cross(b - a, c - a)
                    mag = np.linalg.norm(dir)
                    if mag == 0:
                        norm = [0, 1, 0]
                    else:
                        norm = dir / mag
                    self.normals.data[index] = norm

    @property
    def positions(self):
        return self.attributes.positions

    @positions.setter
    def positions(self, positions):
        self.attributes.positions = positions

    @property
    def normals(self):
        return self.attributes.normals

    @normals.setter
    def normals(self, normals):
        self.attributes.normals = normals

    @property
    def tangents(self):
        return self.attributes.tangents

    @tangents.setter
    def tangents(self, tangents):
        self.attributes.tangents = tangents

    @property
    def texcoords(self):
        return self.attributes.texcoords

    @texcoords.setter
    def texcoords(self, val):
        self.attributes.texcoords = val

    @property
    def colors(self):
        return self.attributes.colors

    @colors.setter
    def colors(self, val):
        self.attributes.colors = val

    @property
    def joints(self):
        return self.attributes.joints

    @joints.setter
    def joints(self, val):
        self.attributes.joints = val

    @property
    def weights(self):
        return self.attributes.weights

    @weights.setter
    def weights(self, val):
        self.attributes.weights = val

    @property
    def indices(self):
        return self._indices

    @indices.setter
    def indices(self, indices):
        if indices is None:
            self._indices = None
            return
        self._indices = indices if isinstance(indices, ScalarArray) else ScalarArray(indices)


class Attributes:
    _positions = None
    _normals = None
    _tangents = None
    texcoords = None
    colors = None
    joints = None
    weights = None

    def __init__(self, attrs=None, gltf=None, positions=None, normals=None, tangents=None, 
                 texcoords=None, colors=None, joints=None, weights=None):
        if attrs:
            if 'POSITION' in attrs:
                self.positions = gltf.accessors[attrs.get('POSITION')]
            if 'NORMAL' in attrs:
                self.normals = gltf.accessors[attrs.get('NORMAL')]
            if 'TANGENT' in attrs:
                self.tangents = gltf.accessors[attrs.get('TANGENT')]
            self.texcoords = []
            if 'TEXCOORD_0' in attrs:
                i = 0
                while True:
                    texcoords = attrs.get('TEXCOORD_' + str(i))
                    if texcoords is None:
                        break
                    self.texcoords.append(gltf.accessors[texcoords])
                    i += 1
            self.colors = []
            if 'COLOR_0' in attrs:
                i = 0
                while True:
                    colors = attrs.get('COLOR_' + str(i))
                    if colors is None:
                        break
                    self.colors.append(gltf.accessors[colors])
                    i += 1
            self.joints = []
            if 'JOINTS_0' in attrs:
                i = 0
                while True:
                    joints = attrs.get('JOINTS_' + str(i))
                    if joints is None:
                        break
                    self.joints.append(gltf.accessors[joints])
                    i += 1
            self.weights = []
            if 'WEIGHTS_0' in attrs:
                i = 0
                while True:
                    weights = attrs.get('WEIGHTS_' + str(i))
                    if weights is None:
                        break
                    self.weights.append(gltf.accessors[weights])
                    i += 1
        else:
            self.positions = positions
            self.normals = normals
            self.tangents = tangents
            self.texcoords = texcoords if texcoords is not None else []
            self.colors = colors if colors is not None else []
            self.joints = joints if joints is not None else []
            self.weights = weights if weights is not None else []

    def render(self, gltf):
        attributes = {}
        if self.positions is not None and self.positions.count:
            attributes['POSITION'] = gltf.index('accessors', self.positions)
        if self.normals is not None and self.normals.count:
            attributes['NORMAL'] = gltf.index('accessors', self.normals)
        if self.tangents is not None and self.tangents.count:
            attributes['TANGENT'] = gltf.index('accessors', self.tangents)
        for i, texcoords in enumerate(self.texcoords):
            attributes['TEXCOORD_' + str(i)] = gltf.index('accessors', texcoords)
        for i, colors in enumerate(self.colors):
            attributes['COLOR_' + str(i)] = gltf.index('accessors', colors)
        for i, joints in enumerate(self.joints):
            attributes['JOINTS_' + str(i)] = gltf.index('accessors', joints)
        for i, weights in enumerate(self.weights):
            attributes['WEIGHTS_' + str(i)] = gltf.index('accessors', weights)
        return attributes

    def sort_joints(self):
        if len(self.joints) > 1:
            joint_set_count = len(self.joints)
            joint_data = np.hstack(tuple(j.data for j in self.joints))
            weight_data = np.hstack(tuple(w.data for w in self.weights))
            sorted_weights = []
            sorted_joints = []
            for w, j in zip(weight_data, joint_data):
                pairs = sorted(zip(w, j), reverse=True)
                new_weights, new_joints = list(zip(*pairs))
                sorted_weights.append(new_weights)
                sorted_joints.append(new_joints)
            sorted_weights = np.array(sorted_weights, 'float32')
            sorted_joints = np.array(sorted_joints, 'uint16')
            self.weights = []
            self.joints = []
            for i in range(joint_set_count):
                self.weights.append(
                    Vec4Array(sorted_weights[:, i * 4:(i + 1) * 4].astype('float32'))
                )
                self.joints.append(
                    Vec4Array(sorted_joints[:, i * 4:(i + 1) * 4].astype('uint16'))
                )

    def __contains__(self, item):
        return (self.positions == item or
                self.normals == item or
                self.tangents == item or
                item in self.texcoords or
                item in self.colors or
                item in self.joints or
                item in self.weights)

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, positions):
        if positions is None:
            self._positions = None
            return
        self._positions = positions if isinstance(positions, Vec3Array) else Vec3Array(positions)

    @property
    def normals(self):
        return self._normals

    @normals.setter
    def normals(self, normals):
        if normals is None:
            self._normals = None
            return
        self._normals = normals if isinstance(normals, Vec3Array) else Vec3Array(normals)

    @property
    def tangents(self):
        return self._tangents

    @tangents.setter
    def tangents(self, tangents):
        if tangents is None:
            self._tangents = None
            return
        self._tangents = tangents if isinstance(tangents, Vec4Array) else Vec4Array(tangents)


class Mesh(BaseGLTFStructure):
    def __init__(self, mesh=None, gltf=None, name=None):
        self.primitives = []
        if mesh:
            self.extras = mesh.get('extras')
            self.name = mesh.get('name')
            for prim in mesh['primitives']:
                self.primitives.append(Primitive(prim, gltf))
            self.weights = mesh.get('weights')
        else:
            self.name = name
            self.extras = None
            self.weights = None

    def __repr__(self):
        return self.name or super(Mesh, self).__repr__()

    def __eq__(self, other):
        # Note that we don't consider `name` when merging nodes
        if not isinstance(other, Mesh):
            return False
        return self.primitives == other.primitives \
            and self.extras == other.extras \
            and self.weights == other.weights

    def add_primitive(self, **kwargs):
        self.primitives.append(Primitive(**kwargs))
        return len(self.primitives) - 1

    def render(self, gltf):
        mesh = {
            'primitives': [primitive.render(gltf) for primitive in self.primitives],
        }
        if self.name:
            mesh['name'] = self.name
        if self.extras:
            mesh['extras'] = self.extras
        if self.weights:
            mesh['weights'] = self.weights

        return mesh

    def repair(self):
        for p in self.primitives:
            p.repair()


class Camera(BaseGLTFStructure):
    perspective = None
    orthographic = None

    def __init__(self, camera=None, name=None, type=None, perspective=None, orthographic=None):
        if camera:
            self.name = camera.get('name')
            self.type = camera.get('type')
            self.perspective = camera.get('perspective')
            self.orthographic = camera.get('orthographic')
        else:
            self.name = name
            if type is None:
                if perspective is not None:
                    type = 'perspective'
                elif orthographic is not None:
                    type = 'orthographic'
                else:
                    raise ValueError('Camera must have type')
            self.type = type
            self.perspective = perspective
            self.orthographic = orthographic

    def render(self):
        camera = {}
        if self.name is not None:
            camera['name'] = self.name
        if self.type:
            camera['type'] = self.type
        if self.type == 'perspective':
            camera['perspective'] = self.perspective
        elif self.type == 'orthographic':
            camera['orthographic'] = self.orthographic
        return camera


class Node(BaseGLTFStructure):
    mesh = None
    translation = None
    rotation = None
    scale = None
    matrix = None
    skin = None
    camera = None
    extensions = None
    extras = None
    weights = None

    def __init__(self, node=None, gltf=None, name=None, mesh=None, children=None,
                 skin=None, camera=None, weights=None):
        if node:
            self.children = []
            self.name = node.get('name')

            if 'mesh' in node:
                self.mesh = gltf.meshes[node['mesh']]

            if 'camera' in node:
                self.camera = gltf.cameras[node['camera']]

            if 'extensions' in node:
                self.extensions = node['extensions']

            if 'extras' in node:
                self.extras = node['extras'].copy()
            
            if 'weights' in node:
                self.weights = node['weights']

            translation = node.get('translation')
            if translation:
                self.translation = np.array(translation, dtype='float32')

            rotation = node.get('rotation')
            if rotation:
                self.rotation = Quaternion(rotation[3], *rotation[:3])

            scale = node.get('scale')
            if scale:
                self.scale = np.array(scale, dtype='float32')

            matrix = node.get('matrix')
            if matrix:
                self.matrix = np.array(matrix, dtype='float32').reshape((4, 4))

            if matrix and (scale or rotation or translation):
                logger.warning('Node defined both matrix and at least one other transform.')
        else:
            self.name = name
            self.mesh = mesh
            self.children = children or []
            self.skin = skin
            self.camera = camera
            self.weights = weights or []

    def __repr__(self):
        return self.name or super(Node, self).__repr__()

    def __eq__(self, other):
        if not (isinstance(other, type(self))
                and (self.mesh, self.name, self.children, self.skin, self.camera, self.weights) ==
                (other.mesh, other.name, other.children, other.skin, other.camera, other.weights)):
            return False

        # quaternions seem to blow up if they're compared to None
        if (self.rotation is None) != (other.rotation is None) or self.rotation != other.rotation:
            return False

        return (np.array_equal(self.translation, other.translation) and
                np.array_equal(self.scale, other.scale) and
                np.array_equal(self.matrix, other.matrix))

    def find_children(self, child_indices, gltf):
        for child_idx in child_indices:
            self.children.append(gltf.nodes[child_idx])

    @property
    def descendancy_has_mesh(self):
        if self.mesh:
            return True
        return any(c.descendancy_has_mesh for c in self.children)

    def get_transform(self, parent_transformation=None, parent_rotation=None):
        transformation = np.identity(4)
        rotation_matrix = parent_rotation
        if self.scale is not None:
            transformation = transformation.dot(
                np.diag(np.append(self.scale, [1]))
            )
        if self.rotation is not None:
            rotation_matrix = (
                self.rotation.inverse.rotation_matrix.dot(rotation_matrix)
                if rotation_matrix is not None
                else self.rotation.inverse.rotation_matrix
            )
            transformation = transformation.dot(
                self.rotation.inverse.transformation_matrix
            )
        if self.translation is not None:
            translation = np.identity(4)
            translation[3, :] += np.append(self.translation, [0])
            transformation = transformation.dot(translation)

        # Only use self.matrix if there is no other transform
        if self.matrix is not None and np.allclose(transformation, np.identity(4)):
            transformation = self.matrix

        if parent_transformation is not None:
            transformation = transformation.dot(parent_transformation)

            scale = [
                np.linalg.norm(transformation[:3, 0]),
                np.linalg.norm(transformation[:3, 1]),
                np.linalg.norm(transformation[:3, 2]),
            ]

            rotation_matrix = transformation[:3, :3] / scale

        return transformation, rotation_matrix

    def apply_transforms(self, parent_transformation=None, parent_rotation=None):
        if not self.descendancy_has_mesh:
            return

        transformation, rotation_matrix = self.get_transform(parent_transformation,
                                                             parent_rotation)

        if self.mesh and not np.allclose(transformation, np.identity(4)):
            for p in self.mesh.primitives:
                if not p.positions:
                    continue
                vertices = p.positions.data
                vertices = np.append(vertices, np.ones([len(vertices), 1]), 1)
                p.positions.data = vertices.dot(transformation)[:, :3].astype('float32')
                if p.normals and rotation_matrix is not None:
                    p.normals.data = p.normals.data.dot(rotation_matrix).astype('float32')

        for child in self.children:
            child.apply_transforms(transformation, rotation_matrix)

        self.translation = None
        self.rotation = None
        self.scale = None
        self.matrix = None

    def render(self, gltf):
        node = {}
        if self.mesh:
            node['mesh'] = gltf.index('meshes', self.mesh)
        if self.camera:
            node['camera'] = gltf.index('cameras', self.camera)
        if self.name:
            node['name'] = self.name
        if self.skin:
            node['skin'] = gltf.index('skins', self.skin)
        if self.children:
            node['children'] = [gltf.index_node(n) for n in self.children]
        if self.translation is not None:
            node['translation'] = self.translation.tolist()
        if self.rotation is not None:
            node['rotation'] = self.rotation.elements.tolist()
            node['rotation'].append(node['rotation'].pop(0))
        if self.scale is not None:
            node['scale'] = self.scale.tolist()
        if self.matrix is not None:
            node['matrix'] = self.matrix.reshape((16,)).tolist()
        if self.extensions:
            node['extensions'] = self.extensions
        if self.extras:
            node['extras'] = self.extras
        if self.weights:
            node['weights'] = self.weights
        return node
