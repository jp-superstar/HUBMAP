#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Surface_mesh_simplification/edge_collapse.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Count_ratio_stop_predicate.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <string>
#include <algorithm> 
using namespace std;

typedef CGAL::Simple_cartesian<double>               Kernel;
typedef Kernel::Point_3                              Point_3;
typedef CGAL::Surface_mesh<Point_3>                  Surface_mesh;
namespace SMS = CGAL::Surface_mesh_simplification;

void saveSimplifiedMesh(string inputSubDirectory, string inputFilename, string outputDirectory, Surface_mesh surface_mesh){
  if (!std::filesystem::exists(outputDirectory)) {
        std::filesystem::create_directory(outputDirectory);
        std::cout << "Directory created.\n";
  }
   string p = "";
   for(int i=inputSubDirectory.size()-1;inputFilename[i]!='/';i--){
        p+=inputFilename[i];
   }
   reverse(p.begin(),p.end()); 
   cout<< p<<endl;
   cout << "Input Sub Directory: " << inputSubDirectory << endl;
   cout<<outputDirectory + '/' +  p<<endl;
   std::filesystem::path dir_path1 = outputDirectory + '/' +  p;


   if (!std::filesystem::exists(dir_path1)) {
        std::filesystem::create_directory(dir_path1);
        std::cout << "inputSubDirectory created.\n";
   }

   string ip = "";
   for(int i=inputFilename.size()-5;inputFilename[i]!='/';i--){
   	ip+=inputFilename[i];
   }   
   reverse(ip.begin(),ip.end());
   cout<< "Output off File name: " << ip <<endl; 
  // string output_path = outputDirectory + '/' +  p + '/' +  ip + ".off";
   // std::filesystem::path dir_path2 = dir_path1 + '/' + ip + ".off";
   cout << "Final Destination Output Directory: " << dir_path1.string() << "/" << ip << ".off" << endl;
   std::string offFileWritePath = dir_path1.string()+"/"+ip+".off";
   CGAL::IO::write_polygon_mesh(offFileWritePath, surface_mesh, CGAL::parameters::stream_precision(17));
   std::cout << "Mesh created in new Location.\n";
        
  

}

void triangulatedMeshSimplification(string inputSubDirectory, string inputFilename, string outputDirectory, double stop_){
  Surface_mesh surface_mesh;
  const std::string filename = CGAL::data_file_path(inputFilename);
  std::ifstream is(filename);
  if(!is || !(is >> surface_mesh))
  {
    std::cerr << "Failed to read input mesh: " << filename << std::endl;
  }
  if(!CGAL::is_triangle_mesh(surface_mesh))
  {
    std::cerr << "Input geometry is not triangulated." << std::endl;
  }
  std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
  // In this example, the simplification stops when the number of undirected edges
  // drops below 10% of the initial count
  double stop_ratio = stop_;
  SMS::Count_ratio_stop_predicate<Surface_mesh> stop(stop_ratio);
  int r = SMS::edge_collapse(surface_mesh, stop);
  std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
  std::cout << "\nFinished!\n" << r << " edges removed.\n" << surface_mesh.number_of_edges() << " final edges.\n";
  std::cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms" << std::endl;
  saveSimplifiedMesh(inputSubDirectory, inputFilename, outputDirectory, surface_mesh);
  //CGAL::IO::write_polygon_mesh(outputFilename, surface_mesh, CGAL::parameters::stream_precision(17));

}

int main(int argc, char** argv)
{
    string sourceInputDirectoryPath = argv[1];
    string sourceOutputDirectoryPath = argv[2];
    double stop_ratio = (argc >= 4) ? std::stod(argv[3]) : 0.1;
    cout<<sourceOutputDirectoryPath<<endl;
    std::cout << "directory_iterator:\n";
    // directory_iterator can be iterated using a range-for loop
    for (auto const& dir_entry : std::filesystem::directory_iterator{sourceInputDirectoryPath}) {
        string subDirectoryPath = dir_entry.path();
	cout<<subDirectoryPath<<endl;
        for (auto const& file_entry : std::filesystem::directory_iterator{subDirectoryPath}){
                string file = file_entry.path();
		cout<<file<<endl;
                triangulatedMeshSimplification(subDirectoryPath, file, sourceOutputDirectoryPath, stop_ratio);
        }
    }

 return EXIT_SUCCESS;
}
     
