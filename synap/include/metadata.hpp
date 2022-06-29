///
/// Network metadata support
///

#pragma once

#include "synap/types.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include <iostream>

namespace synaptics {
namespace synap {

struct TensorAttributes {
    std::string name{};
    DataType dtype{};
    Layout layout{};
    Security security{};
    Shape shape{};
    std::string format{};
    QuantizationScheme qnt_type{};
    int32_t zero_point{};
    float scale{};
    float fl{};
    std::vector<float> mean{};
    float scale2{};
};

struct NetworkMetadata {
    bool valid{};
    bool secure{};
    std::vector<TensorAttributes> inputs;
    std::vector<TensorAttributes> outputs;
};

inline std::ostream& operator<<(std::ostream& os, const TensorAttributes& attr)
{
    os << attr.name << std::endl;
    os << "  dtype:      " << attr.dtype << std::endl;
    os << "  layout:     " << attr.layout << std::endl;
    os << "  shape:      " << attr.shape << std::endl;
    os << "  format:     " << attr.format << std::endl;
    os << "  quantizer:  " << attr.qnt_type << std::endl;
    os << "  zero_pt:    " << attr.zero_point << std::endl;
    os << "  scale:      " << attr.scale << std::endl;
    os << "  fl:         " << attr.fl << std::endl;
    for (auto mean_item: attr.mean) {
        os << "  mean:       " << mean_item << " ";
    }
    os << std::endl;
    os << "  scale2:     " << attr.scale2 << std::endl;
    return os;
}


NetworkMetadata load_metadata(const char* metadata);

}
}  // namespace synaptics
