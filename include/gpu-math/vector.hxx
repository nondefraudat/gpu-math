#pragma once

#include "matrix.hxx"

namespace gpu_math {

using vector = class column_vector;

class column_vector : public matrix {
public:
	column_vector() noexcept : matrix() { }
	column_vector(uint16_t height) noexcept : matrix(height, 1) { }
	column_vector(uint16_t height, float fill_value) noexcept
			: matrix(height, 1, fill_value) { }

	inline float get(uint16_t index) const noexcept {
		return matrix::get(index, 0);
	}
	inline void set(uint16_t index, float value) noexcept {
		return matrix::set(index, 0, value);
	}
};

class row_vector : public matrix {
public:
	row_vector() noexcept : matrix() { }
	row_vector(uint16_t width) noexcept : matrix(width, 1) { }
	row_vector(uint16_t width, float fill_value) noexcept
			: matrix(1, width, fill_value) { }
	
	inline float get(uint16_t index) const noexcept {
		return matrix::get(0, index);
	}
	inline void set(uint16_t index, float value) noexcept {
		return matrix::set(0, index, value);
	}
};

}
