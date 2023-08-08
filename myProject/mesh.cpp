#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Surface_mesh_simplification/edge_collapse.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Count_ratio_stop_predicate.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <string>

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point_3;
typedef CGAL::Surface_mesh<Point_3> Surface_mesh;
namespace SMS = CGAL::Surface_mesh_simplification;

void saveSimplifiedMesh(std::string inputSubDirectory, std::string inputFilename, std::string outputDirectory, Surface_mesh surface_mesh){
    std::filesystem::path outputDirPath(outputDirectory);
    if (!std::filesystem::exists(outputDirectory)) {
        std::filesystem::create_directories(outputDirPath / inputSubDirectory);
        std::cout << "Subdirectory created.\n";
    }

    if (!std::filesystem::exists(outputDirPath / inputSubDirectory)) {
        std::filesystem::create_directories(outputDirPath / inputSubDirectory);
        std::cout << "Subdirectory created.\n";
    }

    std::filesystem::path output_path = outputDirPath / inputSubDirectory / (inputFilename + ".off");
    CGAL::IO::write_polygon_mesh(output_path, surface_mesh, CGAL::parameters::stream_precision(17));
    std::cout << "Mesh created in new Location.\n";
}

void triangulatedMeshSimplification(std::string inputSubDirectory, std::string inputFilename, std::string outputDirectory, double stop){
    Surface_mesh surface_mesh;
    const std::string filename = CGAL::data_file_path(inputFilename);
    std::ifstream is(filename);
    if(!is || !(is >> surface_mesh))
    {
        std::cerr << "Failed to read input mesh: " << filename << std::endl;
        return EXIT_FAILURE;
    }
    if(!CGAL::is_triangle_mesh(surface_mesh))
    {
        std::cerr << "Input geometry is not triangulated." << std::endl;
        return EXIT_FAILURE;
    }

    double stop_ratio = stop;
    SMS::Count_ratio_stop_predicate<Surface_mesh> stop(stop_ratio);
    int r = SMS::edge_collapse(surface_mesh, stop);
    saveSimplifiedMesh(inputSubDirectory, inputFilename, outputDirectory, surface_mesh);
}

int main(int argc, char** argv)
{
    const std::filesystem::path sourceInputDirectoryPath{ argc >= 2 ? argv[1] : std::filesystem::current_path() };
    const std::filesystem::path sourceOutputDirectoryPath{ argc >= 3 ? argv[2] : std::filesystem::current_path() };
    double stop_ratio = (argc >= 4) ? std::stod(argv[3]) : 0.1;

    std::cout << "directory_iterator:\n";
    for (auto const& dir_entry : std::filesystem::directory_iterator{sourceInputDirectoryPath})
    {
        std::string subDirectoryPath = dir_entry.path();
        for (auto const& file_entry : std::filesystem::directory_iterator{subDirectoryPath}){
                std::string file = file_entry.path();
                triangulatedMeshSimplification(subDirectoryPath, file, sourceOutputDirectoryPath.string(), stop_ratio);
        }
    }

    return EXIT_SUCCESS;
}

