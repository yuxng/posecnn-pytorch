#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cfloat>
#include <math.h> 
#include <vector>
#include <ctime>
#include <cstdlib>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cstddef>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/Importer.hpp>
#include <sophus/se3.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/scoped_array.hpp>
#include <pangolin/pangolin.h>

#include "glRender.h"
#include "glRenderTypes.h"
#include "tensor.h"
#include "cudaHelpers.h"
#include "thread_rand.h"

namespace np = boost::python::numpy;

class Synthesizer
{
 public:

  Synthesizer(std::string model_file, std::string pose_file);
  ~Synthesizer();

  void setup(int width, int height);
  void init_rand(unsigned seed);
  void create_window(int width, int height);
  void destroy_window();

  void loadModels(std::string filename);
  void loadPoses(const std::string filename);
  aiMesh* loadTexturedMesh(const std::string filename, std::string & texture_name);

  void initializeBuffers(int model_index, aiMesh* assimpMesh, std::string textureName,
    pangolin::GlBuffer & vertices, pangolin::GlBuffer & canonicalVertices, pangolin::GlBuffer & colors, pangolin::GlBuffer & normals,
    pangolin::GlBuffer & indices, pangolin::GlBuffer & texCoords, pangolin::GlTexture & texture, bool is_textured, int max_vertices);

void render(int width, int height, float fx, float fy, float px, float py, float znear, float zfar, float tnear, float tfar,
            int min_object, int max_object, float std_rotation, float std_translation,
            float* color, float* vertmap, float* class_indexes, 
            float *poses_return, float* centers_return, int is_sampling_object, int is_sampling_pose, int is_display);

  void render_python(np::ndarray const & parameters,
    np::ndarray const & color, np::ndarray const & vertmap, np::ndarray const & class_indexes,
    np::ndarray const & poses, np::ndarray const & center);

 private:
  int counter_;
  int setup_;
  std::string model_file_, pose_file_;

  // poses
  std::vector<float*> poses_;
  std::vector<int> pose_nums_;
  std::vector<bool> is_textured_;
  std::vector<std::vector<Eigen::Quaternionf> > poses_uniform_;
  std::vector<int> pose_index_;

  // 3D models
  std::vector<aiMesh*> assimpMeshes_;

  // pangoline views
  pangolin::View* gtView_;

  // buffers
  std::vector<pangolin::GlBuffer> texturedVertices_;
  std::vector<pangolin::GlBuffer> canonicalVertices_;
  std::vector<pangolin::GlBuffer> vertexColors_;
  std::vector<pangolin::GlBuffer> vertexNormals_;
  std::vector<pangolin::GlBuffer> texturedIndices_;
  std::vector<pangolin::GlBuffer> texturedCoords_;
  std::vector<pangolin::GlTexture> texturedTextures_;

  df::GLRenderer<df::CanonicalVertAndTextureRenderType>* renderer_texture_;
  df::GLRenderer<df::CanonicalVertAndColorRenderType>* renderer_color_;
};

using namespace boost::python;
BOOST_PYTHON_MODULE(libsynthesizer)
{
  np::initialize();
  class_<Synthesizer>("Synthesizer", init<std::string, std::string>())
    .def("setup", &Synthesizer::setup)
    .def("init_rand", &Synthesizer::init_rand)
    .def("render_python", &Synthesizer::render_python)
  ;
}
