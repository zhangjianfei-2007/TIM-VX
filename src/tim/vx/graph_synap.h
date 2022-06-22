#ifndef TIM_VX_GRAPH_SYNAP_H_
#define TIM_VX_GRAPH_SYNAP_H_

#include "graph_private.h"
#include "metadata.hpp"

using namespace synaptics;
using namespace synap;

namespace tim {
namespace vx {

    class GraphSynap : public GraphImpl {
        public:
        GraphSynap(ContextImpl* context, const CompileOption& options = CompileOption::DefaultOptions);
        ~GraphSynap();

        bool Compile() override;
        bool CompileToBinary(void* buf, size_t* size) override;
        bool Run() override;

        private:
            bool isCompiled;
            uint8_t *ebg_buffer{};
            size_t ebg_size;
            struct synaptics::synap::NetworkMetadata meta;
    };

}  // namespace vx
}  // namespace tim
#endif