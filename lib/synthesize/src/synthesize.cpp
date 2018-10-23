#include "synthesize.hpp"

using namespace df;

Synthesizer::Synthesizer(std::string model_file, std::string pose_file)
{
  model_file_ = model_file;
  pose_file_ = pose_file;
  counter_ = 0;
  setup_ = 0;
}

void Synthesizer::setup(int width, int height)
{
  if (setup_ == 0)
  {
    create_window(width, height);

    loadModels(model_file_);
    std::cout << "loaded models" << std::endl;

    loadPoses(pose_file_);
    std::cout << "loaded poses" << std::endl;

    setup_ = 1;
  }
}

void Synthesizer::init_rand(unsigned seed)
{
  ThreadRand::forceInit(seed);
}

Synthesizer::~Synthesizer()
{
  destroy_window();
}

// create window
void Synthesizer::create_window(int width, int height)
{
  pangolin::CreateWindowAndBind("Synthesizer", width, height);
  gtView_ = &pangolin::Display("gt").SetAspect(float(width)/float(height));

  // create render
  renderer_texture_ = new df::GLRenderer<df::CanonicalVertAndTextureRenderType>(width, height);
  renderer_color_ = new df::GLRenderer<df::CanonicalVertAndColorRenderType>(width, height);
}

void Synthesizer::destroy_window()
{
  pangolin::DestroyWindow("Synthesizer");
  delete renderer_texture_;
  delete renderer_color_;
}

// read the poses
void Synthesizer::loadPoses(const std::string filename)
{
  std::ifstream stream(filename);
  std::vector<std::string> model_names;
  std::string name;

  while (std::getline(stream, name))
  {
    std::cout << name << std::endl;
    model_names.push_back(name);
  }
  stream.close();

  // load poses
  const int num_models = model_names.size();
  poses_.resize(num_models);
  pose_nums_.resize(num_models);

  for (int m = 0; m < num_models; ++m)
  {
    // cout lines
    int num_lines = 0;
    std::ifstream stream1(model_names[m]);
    std::string name;

    while ( std::getline (stream1, name) )
      num_lines++;
    stream1.close();
    pose_nums_[m] = num_lines;

    // allocate memory
    float* pose = (float*)malloc(sizeof(float) * num_lines * 7);

    // load data
    FILE* fp = fopen(model_names[m].c_str(), "r");
    for (int i = 0; i < num_lines * 7; i++)
      fscanf(fp, "%f", pose + i);
    fclose(fp);

    poses_[m] = pose;
    std::cout << model_names[m] << std::endl;

    pose_index_.push_back(0);
    std::vector<Eigen::Quaternionf> qv = {};
    poses_uniform_.push_back(qv);
  }

  // sample the poses uniformly
  for (int roll = 0; roll < 360; roll += 15)
  {
    for (int pitch = 0; pitch < 360; pitch += 15)
    {
      for (int yaw = 0; yaw < 360; yaw += 15)
      {
        Eigen::Quaternionf q = Eigen::AngleAxisf(double(roll) * M_PI / 180.0, Eigen::Vector3f::UnitX())
                             * Eigen::AngleAxisf(double(pitch) * M_PI / 180.0, Eigen::Vector3f::UnitY())
                             * Eigen::AngleAxisf(double(yaw) * M_PI / 180.0, Eigen::Vector3f::UnitZ());
        for (int i = 0; i < num_models; i++)
          poses_uniform_[i].push_back(q);
      }
    }
  }
  for (int i = 0; i < num_models; i++)
    std::random_shuffle(poses_uniform_[i].begin(), poses_uniform_[i].end());
  std::cout << poses_uniform_[0].size() << " poses" << std::endl;
}

// read the 3D models
void Synthesizer::loadModels(const std::string filename)
{
  std::ifstream stream(filename);
  std::vector<std::string> model_names;
  std::vector<std::string> texture_names;
  std::string name;

  while (std::getline(stream, name))
  {
    std::cout << name << std::endl;
    model_names.push_back(name);
  }
  stream.close();

  // load meshes
  const int num_models = model_names.size();
  assimpMeshes_.resize(num_models);
  texture_names.resize(num_models);
  int max_vertices = 0;
  for (int m = 0; m < num_models; ++m)
  {
    assimpMeshes_[m] = loadTexturedMesh(model_names[m], texture_names[m]);
    std::cout << texture_names[m] << std::endl;
    if (assimpMeshes_[m]->mNumVertices > max_vertices)
      max_vertices = assimpMeshes_[m]->mNumVertices;
  }

  // buffers
  texturedVertices_.resize(num_models);
  canonicalVertices_.resize(num_models);
  vertexColors_.resize(num_models);
  vertexNormals_.resize(num_models);
  texturedIndices_.resize(num_models);
  texturedCoords_.resize(num_models);
  texturedTextures_.resize(num_models);
  is_textured_.resize(num_models);

  for (int m = 0; m < num_models; m++)
  {
    bool is_textured;
    if (texture_names[m] == "")
      is_textured = false;
    else
      is_textured = true;
    is_textured_[m] = is_textured;

    initializeBuffers(m, assimpMeshes_[m], texture_names[m], texturedVertices_[m], canonicalVertices_[m], vertexColors_[m], vertexNormals_[m],
                      texturedIndices_[m], texturedCoords_[m], texturedTextures_[m], is_textured, max_vertices);
  }
}

aiMesh* Synthesizer::loadTexturedMesh(const std::string filename, std::string & texture_name)
{
  const struct aiScene * scene = aiImportFile(filename.c_str(), aiProcess_JoinIdenticalVertices | aiProcess_GenSmoothNormals);
  if (scene == 0)
    throw std::runtime_error("error: " + std::string(aiGetErrorString()));

  if (scene->mNumMeshes != 1) 
  {
    const int nMeshes = scene->mNumMeshes;
    aiReleaseImport(scene);
    throw std::runtime_error("there are " + std::to_string(nMeshes) + " meshes in " + filename);
  }

  if (!scene->HasMaterials())
    throw std::runtime_error(filename + " has no materials");

  std::cout << scene->mNumMaterials << " materials" << std::endl;

  std::string textureName = filename.substr(0,filename.find_last_of('/')+1);
  for (int i = 0; i < scene->mNumMaterials; ++i) 
  {
    aiMaterial * material = scene->mMaterials[i];
    std::cout << "diffuse: " << material->GetTextureCount(aiTextureType_DIFFUSE) << std::endl;
    std::cout << "specular: " << material->GetTextureCount(aiTextureType_SPECULAR) << std::endl;
    std::cout << "ambient: " << material->GetTextureCount(aiTextureType_AMBIENT) << std::endl;
    std::cout << "shininess: " << material->GetTextureCount(aiTextureType_SHININESS) << std::endl;

    if (material->GetTextureCount(aiTextureType_DIFFUSE)) 
    {
      aiString path;
      material->GetTexture(aiTextureType_DIFFUSE,0,&path);
      textureName = textureName + std::string(path.C_Str());
    }
  }

  aiMesh * assimpMesh = scene->mMeshes[0];
  std::cout << "number of vertices: " << assimpMesh->mNumVertices << std::endl;
  std::cout << "number of faces: " << assimpMesh->mNumFaces << std::endl;

  if (!assimpMesh->HasTextureCoords(0))
    texture_name = "";
  else
    texture_name = textureName;

  return assimpMesh;
}


void Synthesizer::initializeBuffers(int model_index, aiMesh* assimpMesh, std::string textureName,
  pangolin::GlBuffer & vertices, pangolin::GlBuffer & canonicalVertices, pangolin::GlBuffer & colors, pangolin::GlBuffer & normals,
  pangolin::GlBuffer & indices, pangolin::GlBuffer & texCoords, pangolin::GlTexture & texture, bool is_textured, int max_vertices)
{
  std::cout << "number of vertices: " << assimpMesh->mNumVertices << std::endl;
  std::cout << "number of faces: " << assimpMesh->mNumFaces << std::endl;
  vertices.Reinitialise(pangolin::GlArrayBuffer, assimpMesh->mNumVertices, GL_FLOAT, 3, GL_STATIC_DRAW);
  vertices.Upload(assimpMesh->mVertices, assimpMesh->mNumVertices*sizeof(float)*3);

  // normals
  if (assimpMesh->HasNormals())
  {
    normals.Reinitialise(pangolin::GlArrayBuffer, assimpMesh->mNumVertices, GL_FLOAT, 3, GL_STATIC_DRAW);
    normals.Upload(assimpMesh->mNormals, assimpMesh->mNumVertices*sizeof(float)*3);
  }
  else
    throw std::runtime_error("no normals in the mesh");

  // canonical vertices
  std::vector<float3> canonicalVerts(max_vertices);
  for (std::size_t i = 0; i < max_vertices; i++)
  {
    canonicalVerts[i].x = model_index + 1;
    canonicalVerts[i].y = model_index + 1;
    canonicalVerts[i].z = model_index + 1;
  }
  canonicalVertices.Reinitialise(pangolin::GlArrayBuffer, max_vertices, GL_FLOAT, 3, GL_STATIC_DRAW);
  canonicalVertices.Upload(canonicalVerts.data(), max_vertices*sizeof(float3));

  std::vector<uint3> faces3(assimpMesh->mNumFaces);
  for (std::size_t i = 0; i < assimpMesh->mNumFaces; i++) 
  {
    aiFace & face = assimpMesh->mFaces[i];
    if (face.mNumIndices != 3)
      throw std::runtime_error("not a triangle mesh");
    faces3[i] = make_uint3(face.mIndices[0],face.mIndices[1],face.mIndices[2]);
  }

  indices.Reinitialise(pangolin::GlElementArrayBuffer,assimpMesh->mNumFaces*3,GL_UNSIGNED_INT,3,GL_STATIC_DRAW);
  indices.Upload(faces3.data(),assimpMesh->mNumFaces*sizeof(int)*3);

  if (is_textured)
  {
    std::cout << "loading texture from " << textureName << std::endl;
    texture.LoadFromFile(textureName);

    std::cout << "loading tex coords..." << std::endl;
    texCoords.Reinitialise(pangolin::GlArrayBuffer,assimpMesh->mNumVertices,GL_FLOAT,2,GL_STATIC_DRAW);

    std::vector<float2> texCoords2(assimpMesh->mNumVertices);
    for (std::size_t i = 0; i < assimpMesh->mNumVertices; ++i)
      texCoords2[i] = make_float2(assimpMesh->mTextureCoords[0][i].x,1.0 - assimpMesh->mTextureCoords[0][i].y);
    texCoords.Upload(texCoords2.data(),assimpMesh->mNumVertices*sizeof(float)*2);
  }
  else
  {
    // vertex colors
    std::vector<float3> colors3(assimpMesh->mNumVertices);

    for (std::size_t i = 0; i < assimpMesh->mNumVertices; i++) 
    {
      if (assimpMesh->mColors[0])
      {
        aiColor4D & color = assimpMesh->mColors[0][i];
        colors3[i] = make_float3(color.r, color.g, color.b);
      }
      else
      colors3[i] = make_float3(255, 0, 0);
    }
    colors.Reinitialise(pangolin::GlArrayBuffer, assimpMesh->mNumVertices, GL_FLOAT, 3, GL_STATIC_DRAW);
    colors.Upload(colors3.data(), assimpMesh->mNumVertices*sizeof(float)*3);
  }
}


void Synthesizer::render_python(np::ndarray const & parameters,
  np::ndarray const & color, np::ndarray const & vertmap, np::ndarray const & class_indexes,
  np::ndarray const & poses, np::ndarray const & centers)
{

  float* meta = reinterpret_cast<float*>(parameters.get_data());

  int width = int(meta[0]);
  int height = int(meta[1]);
  float fx = meta[2];
  float fy = meta[3];
  float px = meta[4];
  float py = meta[5];
  float znear = meta[6];
  float zfar = meta[7];
  float tnear = meta[8];
  float tfar = meta[9];
  int min_object = int(meta[10]);
  int max_object = int(meta[11]);
  float std_rotation = meta[12];
  float std_translation = meta[13];
  int is_sampling_object = int(meta[14]);
  int is_sampling_pose = int(meta[15]);
  int is_display = int(meta[16]);

  render(width, height, fx, fy, px, py, znear, zfar, tnear, tfar, min_object, max_object, std_rotation, std_translation,
    reinterpret_cast<float*>(color.get_data()), reinterpret_cast<float*>(vertmap.get_data()),
    reinterpret_cast<float*>(class_indexes.get_data()),
    reinterpret_cast<float*>(poses.get_data()), reinterpret_cast<float*>(centers.get_data()),
    is_sampling_object, is_sampling_pose, is_display);
}


void Synthesizer::render(int width, int height, float fx, float fy, float px, float py, float znear, float zfar, float tnear, float tfar,
              int min_object, int max_object, float std_rotation, float std_translation,
              float* color, float* vertmap, float* class_indexes, 
              float *poses_return, float* centers_return, int is_sampling_object, int is_sampling_pose, int is_display)
{
  float threshold = 0.02;
  float std_rot = std_rotation * M_PI / 180.0;
  float std_rot_uniform = 7.5 * M_PI / 180.0;

  pangolin::OpenGlMatrixSpec projectionMatrix_reverse = 
    pangolin::ProjectionMatrixRDF_TopLeft(width, height, fx, -fy, px+0.5, height-(py+0.5), znear, zfar);

  // show gt pose
  glEnable(GL_DEPTH_TEST);
  glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

  // sample the number of objects in the scene
  int num;
  int num_classes = pose_nums_.size();
  std::vector<int> class_ids;

  if (is_sampling_object)
  {
    // sample object classes
    num = irand(min_object, max_object);
    for (int i = 0; i < num; )
    {
      int class_id = irand(0, num_classes);
      int flag = 1;
      for (int j = 0; j < i; j++)
      {
        if(class_id == class_ids[j])
        {
          flag = 0;
          break;
        }
      }
      if (flag)
      {
        class_ids.push_back(class_id);
        i++;
      }
    }
  }
  else
  {
    num = num_classes;
    for (int i = 0; i < num_classes; i++)
      class_ids.push_back(i);
  }

  if (class_indexes)
  {
    for (int i = 0; i < num; i++)
      class_indexes[i] = class_ids[i] + 1;
  }

  // store the poses
  std::vector<Sophus::SE3f> poses(num);

  for (int i = 0; i < num; i++)
  {
    int class_id = class_ids[i];

    while(1)
    {
      // sample a pose
      int seed = irand(0, pose_nums_[class_id]);
      float* pose = poses_[class_id] + seed * 7;

      // translation
      Sophus::SE3f::Point translation;

      if (is_sampling_pose)
      {
        translation(0) = pose[4] + dgauss(0, std_translation);
        translation(1) = pose[5] + dgauss(0, std_translation);
        translation(2) = pose[6] + dgauss(0, std_translation);
      }
      else
      {
        translation(0) = drand(-0.1, 0.1);
        translation(1) = drand(-0.1, 0.1);
        translation(2) = drand(tnear, tfar);
      }

      int flag = 1;
      for (int j = 0; j < i; j++)
      {
        Sophus::SE3f::Point T = poses[j].translation() - translation;
        if (fabs(T(0)) < threshold || fabs(T(1)) < threshold || fabs(T(2)) < 5 * threshold)
        {
          flag = 0;
          break;
        }
      }

      if (flag)
      {
        // quaternion
        Eigen::Quaternionf quaternion;

        if (is_sampling_pose)
        {
          quaternion.w() = pose[0];
          quaternion.x() = pose[1];
          quaternion.y() = pose[2];
          quaternion.z() = pose[3];

          Eigen::Vector3f euler = quaternion.toRotationMatrix().eulerAngles(0, 1, 2);
          euler(0) += dgauss(0, std_rot);
          euler(1) += dgauss(0, std_rot);
          euler(2) += dgauss(0, std_rot);
          Eigen::Quaternionf q = Eigen::AngleAxisf(euler(0), Eigen::Vector3f::UnitX())
                               * Eigen::AngleAxisf(euler(1), Eigen::Vector3f::UnitY())
                               * Eigen::AngleAxisf(euler(2), Eigen::Vector3f::UnitZ());

          quaternion.w() = q.w();
          quaternion.x() = q.x();
          quaternion.y() = q.y();
          quaternion.z() = q.z();
        }
        else
        {
          Eigen::Quaternionf q = poses_uniform_[class_id][pose_index_[class_id]];
          pose_index_[class_id]++;
          if (pose_index_[class_id] >= poses_uniform_[class_id].size())
          {
            pose_index_[class_id] = 0;
            std::random_shuffle(poses_uniform_[class_id].begin(), poses_uniform_[class_id].end());
          }

          Eigen::Vector3f euler = q.toRotationMatrix().eulerAngles(0, 1, 2);
          euler(0) += dgauss(0, std_rot_uniform);
          euler(1) += dgauss(0, std_rot_uniform);
          euler(2) += dgauss(0, std_rot_uniform);
          Eigen::Quaternionf qq = Eigen::AngleAxisf(euler(0), Eigen::Vector3f::UnitX())
                                * Eigen::AngleAxisf(euler(1), Eigen::Vector3f::UnitY())
                                * Eigen::AngleAxisf(euler(2), Eigen::Vector3f::UnitZ());

          quaternion.w() = qq.w();
          quaternion.x() = qq.x();
          quaternion.y() = qq.y();
          quaternion.z() = qq.z();
        }

        const Sophus::SE3f T_co(quaternion, translation);
        poses[i] = T_co;
        if (poses_return)
        {
          poses_return[i * 7 + 0] = quaternion.w();
          poses_return[i * 7 + 1] = quaternion.x();
          poses_return[i * 7 + 2] = quaternion.y();
          poses_return[i * 7 + 3] = quaternion.z();
          poses_return[i * 7 + 4] = translation(0);
          poses_return[i * 7 + 5] = translation(1);
          poses_return[i * 7 + 6] = translation(2);
        }
        break;
      }
    }
  }

  // setup lights
  std::vector<df::Light> lights;

  df::Light spotlight;
  float light_intensity = drand(0.2, 2.0);
  spotlight.position = Eigen::Vector4f(drand(-2, 2), drand(-2, 2), drand(-1, 1), 1);
  spotlight.intensities = Eigen::Vector3f(light_intensity, light_intensity, light_intensity); //strong white light
  spotlight.attenuation = 0.01f;
  spotlight.ambientCoefficient = 0.5f; //no ambient light
  lights.push_back(spotlight);

  int is_texture;
  if (is_textured_[class_ids[0]])
  {
    // render vertmap
    std::vector<Eigen::Matrix4f> transforms(num);
    std::vector<std::vector<pangolin::GlBuffer *> > attributeBuffers(num);
    std::vector<pangolin::GlBuffer*> modelIndexBuffers(num);
    std::vector<pangolin::GlTexture*> textureBuffers(num);
    std::vector<float> materialShininesses(num);

    for (int i = 0; i < num; i++)
    {
      int class_id = class_ids[i];
      transforms[i] = poses[i].matrix().cast<float>();
      materialShininesses[i] = drand(40, 120);
      attributeBuffers[i].push_back(&texturedVertices_[class_id]);
      attributeBuffers[i].push_back(&canonicalVertices_[class_id]);
      attributeBuffers[i].push_back(&texturedCoords_[class_id]);
      attributeBuffers[i].push_back(&vertexNormals_[class_id]);
      modelIndexBuffers[i] = &texturedIndices_[class_id];
      textureBuffers[i] = &texturedTextures_[class_id];
    }

    glClearColor(0, 0, 0, 0);
    renderer_texture_->setProjectionMatrix(projectionMatrix_reverse);
    renderer_texture_->render(attributeBuffers, modelIndexBuffers, textureBuffers, transforms, lights, materialShininesses);
    is_texture = 1;
  }
  else
  {
    // render vertmap
    std::vector<Eigen::Matrix4f> transforms(num);
    std::vector<std::vector<pangolin::GlBuffer *> > attributeBuffers(num);
    std::vector<pangolin::GlBuffer*> modelIndexBuffers(num);
    std::vector<float> materialShininesses(num);

    for (int i = 0; i < num; i++)
    {
      int class_id = class_ids[i];
      transforms[i] = poses[i].matrix().cast<float>();
      materialShininesses[i] = drand(40, 120);
      attributeBuffers[i].push_back(&texturedVertices_[class_id]);
      attributeBuffers[i].push_back(&canonicalVertices_[class_id]);
      attributeBuffers[i].push_back(&vertexColors_[class_id]);
      attributeBuffers[i].push_back(&vertexNormals_[class_id]);
      modelIndexBuffers[i] = &texturedIndices_[class_id];
    }

    glClearColor(0, 0, 0, 0);
    renderer_color_->setProjectionMatrix(projectionMatrix_reverse);
    renderer_color_->render(attributeBuffers, modelIndexBuffers, transforms, lights, materialShininesses);
    is_texture = 0;
  }

  if (vertmap)
  {
    if (is_texture)
      renderer_texture_->texture(0).Download(vertmap, GL_RGB, GL_FLOAT);
    else
      renderer_color_->texture(0).Download(vertmap, GL_RGB, GL_FLOAT);

    // compute object 2D centers
    std::vector<float> center_x(num_classes, 0);
    std::vector<float> center_y(num_classes, 0);
    for (int i = 0; i < num; i++)
    {
      float tx = poses_return[i * 7 + 4];
      float ty = poses_return[i * 7 + 5];
      float tz = poses_return[i * 7 + 6];
      center_x[i] = fx * (tx / tz) + px;
      center_y[i] = fy * (ty / tz) + py;
    }

    if (centers_return)
    {
      for (int i = 0; i < num_classes; i++)
      {
        centers_return[2 * i] = center_x[i];
        centers_return[2 * i + 1] = center_y[i];
      }
    }
  }

  // read color image
  if (color)
  {
    if (is_texture)
      renderer_texture_->texture(1).Download(color, GL_BGR, GL_FLOAT);
    else
      renderer_color_->texture(1).Download(color, GL_BGR, GL_FLOAT);
  }
  std::transform(color, color + height * width * 3, color, [](float p) {return std::min(std::max(float(0.0), p * 255), float(255.0)); });
  
  if (is_display)
  {
    // render color image
    glColor3ub(255,255,255);
    gtView_->ActivateScissorAndClear();
    if (is_texture)
      renderer_texture_->texture(1).RenderToViewportFlipY();
    else
      renderer_color_->texture(1).RenderToViewportFlipY();
    pangolin::FinishFrame();
  }
}
