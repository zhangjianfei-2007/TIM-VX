/****************************************************************************
*
*    Copyright (c) 2020 Vivante Corporation
*
*    Permission is hereby granted, free of charge, to any person obtaining a
*    copy of this software and associated documentation files (the "Software"),
*    to deal in the Software without restriction, including without limitation
*    the rights to use, copy, modify, merge, publish, distribute, sublicense,
*    and/or sell copies of the Software, and to permit persons to whom the
*    Software is furnished to do so, subject to the following conditions:
*
*    The above copyright notice and this permission notice shall be included in
*    all copies or substantial portions of the Software.
*
*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
*    DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/
#include "tim/vx/graph.h"
#include <algorithm>

#include "context_private.h"
#include "graph_private.h"
#include "op_impl.h"
#include "tensor_private.h"
#include "tim/vx/context.h"
#include "tim/vx/ops/nbg.h"
#include "tim/vx/compile_option.h"
#include "vsi_nn_pub.h"

#include "metadata.hpp"
#include "synap/ebg_utils.h"
#include "synap/network.hpp"

#include "graph_synap.h"

using namespace synaptics;
using namespace synap;

namespace tim {
namespace vx {

//graph synap impl
GraphSynap::GraphSynap(ContextImpl* context, const CompileOption& options)
    : GraphImpl(context, options) {}

GraphSynap::~GraphSynap() {}

bool GraphSynap::CompileToBinary(void* buf, size_t* size) {
    return GraphImpl::CompileToBinary(buf, size);
}

bool GraphSynap::Compile() {
    bool status = true;
    size_t bin_size = -1;
    std::vector<uint8_t> nb_buf;

    if (isCompiled)
        return status;

    CompileToBinary(nullptr, &bin_size);
    nb_buf.resize(bin_size);

    //size_t inputs_size = InputsTensor().size();
    //size_t outputs_size = OutputsTensor().size();

    CompileToBinary(nb_buf.data(), &bin_size);

    //call synap nbg to ebg
    ebg_size = nbg_to_ebg(nb_buf.data(), bin_size, &ebg_buffer);
    if (ebg_size == 0 || ebg_buffer == nullptr) {
        VSILOGE("NBG to EBG conversion failed");
        return false;
    }

    // preapare syna meta from tensor spec
    for (std::shared_ptr<Tensor> intensor : InputsTensor()) {
        struct TensorAttributes tensoratrr;
        TensorSpec tspec = intensor->GetSpec();

        //assign tim-vx tensor data to synap tensorattributes
        tensoratrr.dtype = (synaptics::synap::DataType)tspec.datatype_;

        if (tspec.quantization_.Type() == tim::vx::QuantType::ASYMMETRIC)
            tensoratrr.qnt_type = QuantizationScheme::affine_asymmetric;
        else if (tspec.quantization_.Type() == tim::vx::QuantType::NONE)
            tensoratrr.qnt_type = QuantizationScheme::none;
        else
            tensoratrr.qnt_type = QuantizationScheme::dynamic_fixed_point;

        //set scale parameter
        if (tspec.quantization_.Scales().size() > 0)
            tensoratrr.scale = tspec.quantization_.Scales().at(0);
        else if(tspec.quantization_.Scales().size() > 1)
            tensoratrr.scale2 = tspec.quantization_.Scales().at(1);
        tensoratrr.security = synaptics::synap::Security::none;

        //get first input operation
        auto produceop = this->GetProducerOp(InputsTensor().at(0));
        if (produceop != NULL) {
            printf("produce op is existed.\n");
            tim::vx::DataLayout inlayout = produceop->impl()->layout_;
            if (inlayout == DataLayout::CWHN) {  //need offset: cwhn to hnwc
                tensoratrr.format = "nhwc";
                tensoratrr.layout = Layout::nhwc;
            } else if (inlayout == DataLayout::ANY) {
                tensoratrr.format = "none";
                tensoratrr.layout = Layout::none;
            } else if (inlayout == DataLayout::WHCN) { //need offset: whcn to nchw
                tensoratrr.format = "nchw";
                tensoratrr.layout = Layout::nchw;
            } else {
                //does not support in synaptics datalayout
                printf("not supported datalayout %d.\n", (int)inlayout);
                return -1;
            }
            printf("now tensor attr layout info is %d, %d.\n", (int)tensoratrr.layout, (int)inlayout);
        } else {
            printf("use existed tensor.\n");
            //use default
            tensoratrr.format = "nhwc";
            tensoratrr.layout = Layout::nhwc;
        }

        tensoratrr.shape.resize(tspec.shape_.size());
        printf("input shape size %d.\n", tspec.shape_.size());

        int start = 0, end = tspec.shape_.size();
        //update shape info, the input tensor format is always whnc
        for (int32_t dimw : tspec.shape_) {
            if (tensoratrr.layout == Layout::none)
                tensoratrr.shape.at(start) = dimw;
            else  //offset
                tensoratrr.shape.at(end - start - 1) = dimw;
            start++;
        }

        meta.inputs.push_back(tensoratrr);
    }

    printf("input tensor size is %d %d. \n", meta.inputs.size(), OutputsTensor().size());

    for (std::shared_ptr<Tensor> outtensor : OutputsTensor()) {
        struct TensorAttributes tensoratrr;
        TensorSpec tspec = outtensor->GetSpec();

        //assign tim-vx tensor data to synap tensorattributes
        tensoratrr.dtype = (synaptics::synap::DataType)tspec.datatype_;

        if (tspec.quantization_.Type() == tim::vx::QuantType::ASYMMETRIC)
            tensoratrr.qnt_type = QuantizationScheme::affine_asymmetric;
        else if (tspec.quantization_.Type() == tim::vx::QuantType::NONE)
            tensoratrr.qnt_type = QuantizationScheme::none;
        else
            tensoratrr.qnt_type = QuantizationScheme::dynamic_fixed_point;

        printf("1 output shape size %d.\n", tspec.shape_.size());

        //set scale parameter
        if (tspec.quantization_.Scales().size() > 0)
            tensoratrr.scale = tspec.quantization_.Scales().at(0);
        else if(tspec.quantization_.Scales().size() > 1)
            tensoratrr.scale2 = tspec.quantization_.Scales().at(1);
        tensoratrr.security = synaptics::synap::Security::none;

        //get first input operation
        int outtensorsize = OutputsTensor().size();
        printf("outtensorsize %d.\n", outtensorsize);
        std::shared_ptr<Operation> comsumeop = NULL;
        if (this->GetConsumersOp(OutputsTensor().at(outtensorsize - 1)).size() != 0)
            comsumeop = this->GetConsumersOp(OutputsTensor().at(outtensorsize - 1)).at(0);
        if (comsumeop) {
            printf("comsume op existed.\n");
            tim::vx::DataLayout outlayout = comsumeop->impl()->layout_;
            if (outlayout == DataLayout::CWHN) {  //need offset: cwhn to hnwc
                tensoratrr.format = "nhwc";
                tensoratrr.layout = Layout::nhwc;
            } else if (outlayout == DataLayout::ANY) {
                tensoratrr.format = "none";
                tensoratrr.layout = Layout::none;
            } else if (outlayout == DataLayout::WHCN) { //need offset: whcn to nchw
                tensoratrr.format = "nchw";
                tensoratrr.layout = Layout::nchw;
            } else {
                //does not support in synaptics datalayout
                printf("not supported datalayout %d.\n", (int)outlayout);
                return -1;
            }
            printf("output tensor attr layout info is %d, %d.\n", (int)tensoratrr.layout, (int)outlayout);
        } else {
            tensoratrr.format = "none";
            tensoratrr.layout = Layout::none;
        }

        tensoratrr.shape.resize(tspec.shape_.size());
        printf("output shape size %d.\n", tspec.shape_.size());

        int start = 0, end = tspec.shape_.size();
        //update shape info, the input tensor format is always whnc
        for (int32_t dimw : tspec.shape_) {
            if (tensoratrr.layout == Layout::none)
                tensoratrr.shape.at(start) = dimw;
            else  //offset
                tensoratrr.shape.at(end - start - 1) = dimw;
            start++;
        }
        meta.outputs.push_back(tensoratrr);
    }
    printf("output tensor size is %d. \n", meta.outputs.size());
    isCompiled = true;

    return status;
}

bool GraphSynap::Run() {
    //call ebg to run
    Network synap_net;
    //Preprocessor preprocessor;

    Compile();
    //the rick here is input and output data
    synap_net.load_model(ebg_buffer, ebg_size, &meta);

    // Select allocator for the buffers
    synaptics::synap::Allocator* allocator = std_allocator();
    if (!allocator) {
        printf("Error, selected memory allocation method not available.\n");
        return 1;
    }

    // Set input tensors allocator
    for (synaptics::synap::Tensor& intensor : synap_net.inputs) {
        printf("Input buffer: %s size: %d.\n", intensor.name().c_str(), intensor.size());
        intensor.buffer()->set_allocator(allocator);
    }

    // Set output tensors allocator
    for (synaptics::synap::Tensor& outtensor : synap_net.outputs) {
        printf("Output buffer: %s size: %d.\n", outtensor.name().c_str(), outtensor.size());
        outtensor.buffer()->set_allocator(allocator);
    }

    printf("call synap run predict function %d, %d.\n", synap_net.inputs.size(), synap_net.outputs.size());

    //add data into synap_net
    int i = 0;
    for (synaptics::synap::Tensor& synap_in: synap_net.inputs) {
        const auto& t_fmt = synap_in.format();
        Dimensions t_dim = synap_in.dimensions();

        printf("data info format %d, %d, %d, %d, input size %d, %d.\n", t_dim.w, t_dim.h, t_dim.c, t_dim.n, synap_in.size(), InputsTensor().size());

        //copy data to synaptics tensor
        if (synap_in.data() && InputsTensor().at(i))
            InputsTensor().at(i)->CopyDataFromTensor(synap_in.data());

        i++;
    }

    bool success = synap_net.predict();

    printf("the inference result is %d. \n", success);

    //check output
    i = 0;
    for (synaptics::synap::Tensor& synap_out: synap_net.outputs) {
        const auto& t_fmt = synap_out.format();
        Dimensions t_dim = synap_out.dimensions();

        printf("data info format %d, %d, %d, %d, output size %d, %d.\n", t_dim.w, t_dim.h, t_dim.c, t_dim.n, synap_out.size(), OutputsTensor().size());

        //copy data to synaptics tensor
        if (synap_out.data())
            OutputsTensor().at(i)->CopyDataToTensor(synap_out.data());

        //for (int j = 0; j < synap_out.size(); j++)
        printf("result:  %f \n", *synap_out.as_float());

        i++;
    }

    return success;
}

}  // namespace vx
}  // namespace tim
