/*
 * NDA AND NEED-TO-KNOW REQUIRED
 *
 * Copyright (C) 2013-2020 Synaptics Incorporated. All rights reserved.
 *
 * This file contains information that is proprietary to Synaptics
 * Incorporated ("Synaptics"). The holder of this file shall treat all
 * information contained herein as confidential, shall use the
 * information only for its intended purpose, and shall not duplicate,
 * disclose, or disseminate any of this information in any manner
 * unless Synaptics has otherwise provided express, written
 * permission.
 *
 * Use of the materials may require a license of intellectual property
 * from a third party or from Synaptics. This file conveys no express
 * or implied licenses to any intellectual property rights belonging
 * to Synaptics.
 *
 * INFORMATION CONTAINED IN THIS DOCUMENT IS PROVIDED "AS-IS", AND
 * SYNAPTICS EXPRESSLY DISCLAIMS ALL EXPRESS AND IMPLIED WARRANTIES,
 * INCLUDING ANY IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE, AND ANY WARRANTIES OF NON-INFRINGEMENT OF ANY
 * INTELLECTUAL PROPERTY RIGHTS. IN NO EVENT SHALL SYNAPTICS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, PUNITIVE, OR
 * CONSEQUENTIAL DAMAGES ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OF THE INFORMATION CONTAINED IN THIS DOCUMENT, HOWEVER CAUSED AND
 * BASED ON ANY THEORY OF LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * NEGLIGENCE OR OTHER TORTIOUS ACTION, AND EVEN IF SYNAPTICS WAS
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. IF A TRIBUNAL OF
 * COMPETENT JURISDICTION DOES NOT PERMIT THE DISCLAIMER OF DIRECT
 * DAMAGES OR ANY OTHER DAMAGES, SYNAPTICS' TOTAL CUMULATIVE LIABILITY
 * TO ANY PARTY SHALL NOT EXCEED ONE HUNDRED U.S. DOLLARS.
 */
///
/// Synap data tensor.
///

#pragma once

#include "synap/buffer.hpp"
#include "synap/types.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <string>
#include <memory>
#include <vector>

namespace synaptics {
namespace synap {

class NetworkPrivate;
class TensorAttributes;

/// Synap data tensor.
/// It's not possible to create tensors outside a Network,
/// users can only access tensors created by the Network itself.
class Tensor {
public:
    // In/out type
    enum class Type { none, in, out };

    // Constructor.
    Tensor(NetworkPrivate* np, int32_t ix, Type ttype, const TensorAttributes* attr);

    // Move constructor
    Tensor(Tensor&& rhs) noexcept;

    ~Tensor();
 
    /// Get the name of the tensor.
    /// Can be useful in case of networks with multiple inputs or outputs
    /// to identify a tensor with a string instead of a positional index.
    /// @return tensor name
    const std::string& name() const;

    /// Get the Shape of the Tensor, that is the number of elements in each dimension.
    /// The order of the dimensions is specified by the tensor layout.
    /// @return tensor shape
    const Shape& shape() const;

    /// Get the Dimensions of the Tensor, that is the number of elements in each dimension.
    /// The returned values are independent of the tensor layout.
    /// @return tensor dimensions (all 0s if the rank of the tensor is not 4)
    const Dimensions dimensions() const;


    /// Get the layout of the Tensor, that is how data are organized in memory.
    /// SyNAP supports two layouts: `NCHW` and `NHWC`.
    /// The *N* dimension (number of samples) is present for
    /// compatibility with standard conventions but must always be one.
    /// @return tensor layout
    Layout layout() const;

    /// Get the format of the Tensor, that is a description of what the data represents.
    /// This is a free-format string whose meaning is application dependent, for example
    /// "rgb", "bgr".
    /// @return tensor format
    std::string format() const;

    /// Get tensor data type.
    /// The integral types are used to represent quantized data. The details of the quantization
    /// parameters and quantization scheme are not directly available, an user can use quantized
    /// data by converting them to 32-bits *float* using the `as_float()` method below
    /// @return the type of each item in the tensor.
    DataType data_type() const;

    /// Get tensor security attribute.
    /// @return security attribute of the tensor (none if the model is not secure).
    Security security() const;
    
    /// @return size *in bytes* of the tensor data
    size_t size() const;

    /// Get the number of items in the tensor.
    /// A tensor `size()` is alwas equal to `item_count()`  multiplied by the size of
    /// the tensor data type.
    /// @return number of data items in the tensor
    size_t item_count() const;

    /// Copy the content of a vector to the tensor data buffer.
    /// The data is considered as raw data so no type conversion is done whatever the actual
    /// data-type of the tensor. The size of the vector must be equal to the `size()`
    /// of the tensor.
    /// @param data: data to be copied in tensor. Data size must match tensor size.
    /// @return true if success
    bool assign(const std::vector<uint8_t>& data);

    /// Copy data to the tensor data buffer.
    /// The data is considered as raw data so no type conversion is done whatever the actual
    /// data-type of the tensor. The data size must be equal to the `size()`
    /// of the tensor.
    /// @param data: pointer to data to be copied
    /// @param size: size of data to be copied
    /// @return true if success
    bool assign(const void* data, size_t size);
    
    /// Copy the content of a tensor to the tensor data buffer.
    /// No type conversion is done, the data type and size of the two tensors must match.
    /// @param src: source tensor containing the data to be copied.
    /// @return true if success, false if type or size mismatch
    bool assign(const Tensor& src);

    /// Get a pointer to the beginning of the data inside the tensor data buffer, if any.
    /// The method returns a `void` pointer since the actual data type is what returned by
    /// the `data_type()` method.
    /// Please note that this is a pointer to the data inside the tensor data buffer:
    /// this means that the returned pointer must *not* be freed, memory will be released
    /// automatically when the tensor data buffer is destroyed.
    /// @return pointer to the raw data inside the data buffer, nullptr if none.
    void* data();
    const void* data() const;

    /// Get a pointer to the tensor content converted to float.
    /// The method always returns a `float` pointer. If the actual data type of the tensor
    /// is not float, the conversion is performed interally, so the user doesn't have to care about
    /// how the data are internally represented.
    ///
    /// Please note that this is a pointer to floating point data inside the tensor itself:
    /// this means that the returned pointer must *not* be freed, memory will be released
    /// automatically when the tensor is destroyed.
    /// @return pointer to float[item_count()] array representing tensor content converted to float
    ///         (nullptr if tensor has no data)
    const float* as_float() const;

    /// Get pointer to the tensor's current data Buffer if any.
    /// This will be the default buffer of the tensor unless the user has assigned a different
    /// buffer using set_buffer()
    /// @return current data buffer, or nullptr if none
    Buffer* buffer();

    /// Set the tensor's current data buffer.
    /// The buffer size must match the tensor size or the operation will be rejected.
    /// Normally the provided buffer should live at least as long as the tensor itself.
    /// If the buffer object is destroyed before the tensor, it will be automatically unset
    /// and the tensor will remain buffer-less.
    /// @param buffer: buffer to be used for this tensor.
    ///                The buffer size must match the tensor size.
    /// @return true if success
    bool set_buffer(Buffer* buffer);


private:
    /// Private implementation details
    struct Private;
    std::unique_ptr<Private> d;
};


/// Tensor collection
class Tensors {
public:
    /// Construct from a vector of tensors
    Tensors(std::vector<Tensor>& tensors) : _tensors{tensors} {}

    /// Access/iteration methods
    size_t size() const { return _tensors.size(); }
    const Tensor& operator[](size_t index) const;
    Tensor& operator[](size_t index)
    {
        return const_cast<Tensor&>(static_cast<const Tensors&>(*this)[index]);
    }
    std::vector<Tensor>::iterator begin() { return _tensors.begin(); }
    std::vector<Tensor>::iterator end() { return _tensors.end(); }

private:
    std::vector<Tensor>& _tensors;
};


}  // namespace synap
}  // namespace synaptics
