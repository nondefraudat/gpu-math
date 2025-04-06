#include "gpu-math/Vector.hxx"

namespace gm {

// Device methods declaration

__global__ void _add(float source[], float target[], uint32_t size);
__global__ void _subtract(float source[], float target[], uint32_t size);

// Mutators

void Vector::add(const Vector &vector) {
	_size_test(vector.size());
	_add<<<size(), 1>>>(_device_data(),
			vector._device_data(), size());
}

void Vector::subtract(const Vector &vector) {
	_size_test(vector.size());
	_subtract<<<size(), 1>>>(_device_data(),
			vector._device_data(), size());
}

// Operators

Vector &Vector::operator=(real_t value) {
	fill(value); return *this;
}

Vector &Vector::operator+=(const Vector &target) {
	add(target); return *this;
}

Vector Vector::operator+(const Vector &target) const {
	_size_test(target.size());
	Vector result(*this);
	result.add(target);
	return result;
}

Vector &Vector::operator-=(const Vector &target) {
	subtract(target); return *this;
}

Vector Vector::operator-(const Vector &target) const {
	_size_test(target.size());
	Vector result(*this);
	result.subtract(target);
	return result;
}

real_t Vector::operator[](size_t index) const {
	_index_test(index);
	return _host_buffer()[index];
}

std::ostream &operator<<(std::ostream &stream, const Vector &vector) {
	stream << vector[0];
	for (uint32_t i = 1; i < vector.size(); i++) {
		stream << " " << vector[i];
	}
	return stream;
}

// Device method definitions

__global__ void _add(float source[], float target[], uint32_t size) {
	uint32_t index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < size) {
		source[index] += target[index];
	}
}

__global__ void _subtract(float source[], float target[], uint32_t size) {
	uint32_t index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < size) {
		source[index] -= target[index];
	}
}

}
