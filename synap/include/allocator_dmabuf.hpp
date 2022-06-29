/*
 * NDA AND NEED-TO-KNOW REQUIRED
 *
 * Copyright (C) 2021 Synaptics Incorporated. All rights reserved.
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

#include "synap/allocator.hpp"


namespace synaptics {
namespace synap {


class AllocatorDmabuf : public Allocator {
public:
    AllocatorDmabuf(bool contiguous, bool secure);
    ~AllocatorDmabuf();
    Memory alloc(size_t size) override;
    void dealloc(const Memory& mem) override;
    bool cache_flush(const Memory& mem, size_t size) override;
    bool cache_invalidate(const Memory& mem, size_t size) override;

    Memory do_alloc(size_t size, uint32_t heap_mask);
    bool available() const override { return _available; }

protected:
    static bool suspend_cpu_access(int fd);
    static bool resume_cpu_access(int fd);
    bool init();
    uint32_t heap_mask() const { return _heap_mask_std; }

private:
    bool _available{};
    int _ion_fd{-1};
    bool _contiguous{};
    bool _secure{};
    uint32_t _heap_mask_std{};
};


}  // namespace synap
}  // namespace synaptics
