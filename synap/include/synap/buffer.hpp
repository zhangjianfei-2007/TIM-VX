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
/// Synap data buffer.
///

#pragma once

#include "synap/allocator.hpp"

#include <memory>
#include <vector>


namespace synaptics {
namespace synap {

class BufferPrivate;


/// Synap data buffer.
class Buffer {
public:
    /// Create an empty data buffer.
    /// @param allocator: allocator to be used (default is malloc-based)
    Buffer(Allocator* allocator = nullptr);

    /// Create and allocate a data buffer.
    /// @param size: buffer size
    /// @param allocator: allocator to be used (default is malloc-based)
    Buffer(size_t size, Allocator* allocator = nullptr);

    /// Create a data buffer to refer to an existing memory area.
    /// The user must make sure that the provided memory is correctly aligned and padded.
    /// The specified memory area will *not* be deallocated when the buffer is destroyed.
    /// It is the responsiblity of the caller to release mem_id *after* the Buffer has been
    /// destroyed.
    /// @param mem_id: id of an existing dmabuf (or part of it) registered with the TZ kernel.
    /// @param offset: offset of the actual data insithe the memory area
    /// @param size: size of the actual data 
    Buffer(uint32_t mem_id, size_t offset, size_t size);

    // Move constructor
    Buffer(Buffer&& rhs) noexcept;

    // Deallocate buffer.
    ~Buffer();

    /// Resize buffer.
    /// Only possible if an allocator was provided. Any previous content is lost.
    ///
    /// @param size: new buffer size
    /// @return true if success
    bool resize(size_t size);

    /// Copy data in buffer.
    /// Always successful if the input data size is the same as current buffer size,
    /// otherwise the buffer is resized if possible.
    ///
    /// @param data: data to be copied
    /// @return true if all right
    bool assign(const std::vector<uint8_t>& data);

    /// Copy data in buffer.
    /// Always successful if the input data size is the same as current buffer size,
    /// otherwise the buffer is resized if possible.
    ///
    /// @param data: pointer to data to be copied
    /// @param size: size of data to be copied
    /// @return true if all right
    bool assign(const void* data, size_t size);

    /// Actual data size
    size_t size() const;

    /// Actual data
    const void* data() const;
    void* data();

    /// Enable/disable the possibility for the CPU to read/write the buffer data.
    /// By default CPU access to data is enabled.
    /// CPU access can be disabled in case the CPU doesn't need to read or write
    /// the buffer data and can provide some performance improvements when
    /// the data is only generated/used by another HW components.
    ///
    /// @note reading or writing buffer data while CPU access is disabled might
    /// cause loss or corruption of the data in the buffer.
    ///
    /// @param allow: false to indicate the CPU will not access buffer data
    /// @return current setting
    bool allow_cpu_access(bool allow);

    /// Change the allocator.
    /// Can only be done if the buffer is empty.
    /// @param allocator: allocator
    /// @return true if success
    bool set_allocator(Allocator* allocator);

    // @return file descriptor id of associated memory block if any
    int32_t memory_fd() const;

    // @return mem_id of associated memory block if any
    uint32_t mem_id() const;

    // @return bid of associated memory block if any
    uint32_t bid() const;

    // @return private implementation details
    BufferPrivate* priv() { return d.get(); }

private:
    // Private implementation details
    std::unique_ptr<BufferPrivate> d;
};


}  // namespace synap
}  // namespace synaptics
