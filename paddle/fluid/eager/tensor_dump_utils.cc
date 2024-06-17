// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/eager/tensor_dump_utils.h"

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/api/ext/tensor_compat.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/selected_rows.h"
#include "paddle/phi/core/tensor_utils.h"

namespace egr {

static std::once_flag dump_list_init_flag;

static std::unordered_set<std::string>& api_name_dump_list() {
  static std::unordered_set<std::string> _api_name_dump_list = {};
  return _api_name_dump_list;
}

static void InitDumpListFormEnv() {
  api_name_dump_list();
  // export PADDLE_DUMP_OP_LIST="all"
  // export PADDLE_DUMP_OP_LIST="batch_norm,batch_norm_grad"
  // export PADDLE_DUMP_OP_LIST="batch_norm,conv2d"
  const char* op_type_dump = std::getenv("PADDLE_DUMP_OP_LIST");

  if (op_type_dump) {
    std::stringstream ss(op_type_dump);
    std::string op_type;
    while (std::getline(ss, op_type, ',')) {
      api_name_dump_list().emplace(op_type);
    }
  }
  for (auto const& key : api_name_dump_list()) {
    LOG(INFO) << "PADDLE_DUMP_OP_LIST: " << key;
  }
}

bool SkipDump(const std::string& api_name) {
  if (api_name_dump_list().count("all") != 0) return false;
  if (api_name_dump_list().count(api_name) != 0) return false;
  return true;
}

void DumpTensorToFile(const std::string& api_unique,
                      const std::string& api_name,
                      const std::string& arg_type,
                      const std::string& arg_name,
                      const Tensor& tensor) {
  std::call_once(dump_list_init_flag, InitDumpListFormEnv);

  if (SkipDump(api_name)) return;

  if (tensor.initialized()) {
    auto& tensor_name = tensor.name();
    const phi::DenseTensor* dense_tensor{nullptr};
    if (tensor.is_dense_tensor()) {
      dense_tensor = static_cast<const phi::DenseTensor*>(tensor.impl().get());
    } else if (tensor.is_selected_rows()) {
      dense_tensor = &(
          static_cast<const phi::SelectedRows*>(tensor.impl().get())->value());
    } else {
      VLOG(10) << "Only DenseTensor or SelectedRows need to check, "
               << tensor_name << " is no need.";
      return;
    }

    std::string adr_name = paddle::string::Sprintf("%d", tensor.impl());
    tensor_dump(
        api_unique, api_name, arg_type, arg_name, adr_name, *dense_tensor);
  }
}

void DumpTensorToFile(const std::string& api_unique,
                      const std::string& api_name,
                      const std::string& arg_type,
                      const std::string& arg_name,
                      const paddle::optional<Tensor>& tensor) {
  if (tensor) {
    DumpTensorToFile(api_unique, api_name, arg_type, arg_name, *tensor);
  }
}

void DumpTensorToFile(const std::string& api_unique,
                      const std::string& api_name,
                      const std::string& arg_type,
                      const std::string& arg_name,
                      const std::vector<Tensor>& tensors) {
  for (auto& tensor : tensors) {
    DumpTensorToFile(api_unique, api_name, arg_type, arg_name, tensor);
  }
}

void DumpTensorToFile(const std::string& api_unique,
                      const std::string& api_name,
                      const std::string& arg_type,
                      const std::string& arg_name,
                      const paddle::optional<std::vector<Tensor>>& tensors) {
  if (tensors) {
    DumpTensorToFile(api_unique, api_name, arg_type, arg_name, *tensors);
  }
}

template <typename T>
inline std::string GetTensorDesc(const std::string& adr_name,
                                 const phi::DenseTensor& tensor) {
  std::string dtype_str = phi::DataTypeToString(tensor.dtype());
  std::stringstream ss;
  ss << "Name: " << adr_name;
  if (tensor.initialized()) {
    ss << ", initialized: 1, place: " << tensor.place()
       << ", dtype: " << dtype_str << ", format: " << tensor.layout()
       << ", dims: [" << tensor.dims() << "]"
       << ", capacity: <" << tensor.capacity() << ">";
  } else {
    ss << ", initialized: 0, place: Unknown"
       << ", dtype: " << dtype_str << ", format: " << tensor.layout()
       << ", dims: [" << tensor.dims() << "]";
  }
  return ss.str();
}

void DumpTensorDesc(const std::string& fname, const std::string& desc) {
  std::ofstream fout(fname);
  PADDLE_ENFORCE_EQ(static_cast<bool>(fout),
                    true,
                    phi::errors::NotFound("Cannot open %s to write", fname));
  fout << "TensorDesc = { " << desc << " }\n";  // desc
  fout.close();
}

template <typename T>
void DumpTensorData(const std::string& fname, const std::vector<T>& data) {
  std::ofstream fout(fname, std::ios::out | std::ofstream::binary);
  PADDLE_ENFORCE_EQ(static_cast<bool>(fout),
                    true,
                    phi::errors::NotFound("Cannot open %s to write", fname));
  fout.write(reinterpret_cast<const char*>(&data[0]), data.size() * sizeof(T));
  fout.close();
}

template <typename T>
void TensorDumpVisitor::apply(
    typename std::enable_if<
        std::is_floating_point<T>::value ||
        std::is_same<T, ::paddle::platform::complex<float>>::value ||
        std::is_same<T, ::paddle::platform::complex<double>>::value>::type*)
    const {
  auto tensor_desc = GetTensorDesc<T>(adr_name, tensor);

  std::vector<T> tensor_data;

  auto* dev_ctx =
      paddle::platform::DeviceContextPool::Instance().Get(tensor.place());
  if (tensor.place() == phi::CPUPlace()) {
    dev_ctx = static_cast<phi::CPUContext*>(
        paddle::platform::DeviceContextPool::Instance().Get(tensor.place()));
    phi::TensorToVector(tensor, *dev_ctx, &tensor_data);
  } else {
    dev_ctx = static_cast<paddle::platform::CustomDeviceContext*>(
        paddle::platform::DeviceContextPool::Instance().Get(tensor.place()));

    phi::DenseTensor cpu_tensor;
    phi::Copy(*dev_ctx, tensor, phi::CPUPlace(), true, &cpu_tensor);
    phi::TensorToVector(cpu_tensor, *dev_ctx, &tensor_data);
  }

  std::string folder_path = "tensor_dump/" + api_unique + "/" + arg_type + "/";
  std::string mkdir_cmd = "mkdir -p " + folder_path;
  PADDLE_ENFORCE_EQ(system(mkdir_cmd.c_str()),
                    0,
                    paddle::platform::errors::NotFound(
                        "Cannot create folder %s", folder_path));

  std::string file_path = folder_path + arg_name + "_" + adr_name;

  VLOG(1) << "Dumping kernel<" << api_name << "> tensor <" << adr_name
          << "> to file: " << file_path << ".txt";
  DumpTensorDesc(file_path + ".txt", tensor_desc);
  DumpTensorData(file_path + ".bin", tensor_data);
}

void tensor_dump(const std::string& api_unique,
                 const std::string& api_name,
                 const std::string& arg_type,
                 const std::string& arg_name,
                 const std::string& adr_name,
                 const phi::DenseTensor& tensor) {
  TensorDumpVisitor vistor(
      api_unique, api_name, arg_type, arg_name, adr_name, tensor);
  paddle::framework::VisitDataType(
      paddle::framework::TransToProtoVarType(tensor.dtype()), vistor);
}

}  // namespace egr