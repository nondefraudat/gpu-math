#include "matrix.hxx"

namespace gpu_math {

__global__ void fill(float data[], uint32_t size, float value);
__global__ void multiply(float source[], float target[], float destination[],
		uint16_t source_height, uint16_t source_width, uint16_t target_widht);
__global__ void add(float source[], float target[],
		float destination[], uint32_t size);

void matrix::fill(float value) noexcept {
	gpu_math::fill<<<size(), 1>>>(device_data(), size(), value);
}

matrix matrix::multiply(const matrix &other) const noexcept {
	if (width() != other.height()) {
		return matrix(1, 1, std::numeric_limits<float>::quiet_NaN());
	}
	matrix result(height(), other.width(), 0.f);
	gpu_math::multiply<<<{result.height(), result.width()}, 1>>>(
			device_data(), other.device_data(), result.device_data(),
			height(), width(), other.width());
	return result;
}

matrix matrix::add(const matrix &other) const noexcept {
	if (height() != other.height() || width() != other.width()) {
		return matrix(1, 1, std::numeric_limits<float>::quiet_NaN());
	}
	matrix result(height(), width(), 0.f);
	gpu_math::add<<<size(), 1>>>(device_data(), other.device_data(),
			result.device_data(), size());
	return result;
}

std::ostream &operator<<(std::ostream &os, const matrix &m) {
	for (uint16_t i = 0; i < m.height(); i++) {
		os << m.get(i, 0);
		for (uint16_t j = 1; j < m.width(); j++) {
			os << ' ' << m.get(i, j);
		}
		os << '\n';
	}
	return os;
}

__global__ void fill(float data[], uint32_t size, float value) {
	uint32_t index = blockIdx.x;
	if (index < size) {
		data[index] = value;
	}
}

__global__ void multiply(float source[], float target[], float destination[],
		uint16_t source_height, uint16_t source_width, uint16_t target_widht) {
	if ( !(blockIdx.x < source_height && blockIdx.y < target_widht) ) {
		return;
	}
	uint32_t destination_index = blockIdx.x*target_widht + blockIdx.y;
	uint32_t source_start_index = blockIdx.x*source_width;
	uint32_t target_start_column = blockIdx.y;
	for (uint32_t i = 0; i < source_width; i++) {
		uint32_t source_index = source_start_index + i;
		uint32_t target_index = i*target_widht + target_start_column;
		destination[destination_index] +=
				source[source_index]*target[target_index];
	}
}

__global__ void add(float source[], float target[],
		float destination[], uint32_t size) {
	uint32_t index = blockIdx.x;
	if (index < size) {
		destination[index] = source[index] + target[index];
	}
}
}
