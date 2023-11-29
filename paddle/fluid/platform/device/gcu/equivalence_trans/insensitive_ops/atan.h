/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <memory>
#include "paddle/fluid/platform/device/gcu/register/register.h"

namespace paddle {
namespace platform {
namespace gcu {
const char *const kAtan = "atan";
const char *const kAtanGrad = "atan_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, AtanEquivalenceTrans) {
  auto result = builder::Atan(*(map_inputs["X"].at(0)));
  return std::make_shared<GcuOp>(result);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, AtanGradEquivalenceTrans) {
  GcuOp out = *(map_inputs["X"].at(0));
  GcuOp out_grad = *(map_inputs["Out@GRAD"].at(0));

  auto param = builder::OnesLike(out);
  return std::make_shared<GcuOp>(out_grad * param / (param + out * out));
}

EQUIVALENCE_TRANS_FUNC_REG(kAtan, INSENSITIVE, AtanEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kAtanGrad, INSENSITIVE, AtanGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
