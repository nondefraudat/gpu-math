#pragma once

#include "wandering_data.hxx"
#include <limits>
#include <iostream>

namespace gpu_math {

class matrix : public wandering_data {
public:
	matrix() noexcept : wandering_data() { }
	matrix(uint16_t height, uint16_t width) noexcept
			: wandering_data(height, width) { }
	matrix(uint16_t height, uint16_t width, float fill_value) noexcept
			: matrix(height, width) { fill(fill_value); }

	inline float get(uint16_t row, uint16_t column) const noexcept {
		return get(index(row, column));
	}

	inline void set(uint16_t row, uint16_t column, float value) noexcept {
		set(index(row, column), value);
	}

	void fill(float value) noexcept;
	matrix multiply(const matrix &other) const noexcept;
	matrix add(const matrix &other) const noexcept;

	inline matrix operator*(const matrix &other) const noexcept {
		return multiply(other);
	}	
	inline matrix operator+(const matrix &other) const noexcept {
		return add(other);
	}

	friend std::ostream &operator<<(std::ostream& os, const matrix &m);

protected:
	inline uint32_t index(uint16_t row, uint16_t column) const noexcept {
		return static_cast<uint32_t>(row)*width() + column;
	}

	inline float get(uint32_t index) const noexcept {
		if (index < size()) {
			return host_data()[index];
		}
		return std::numeric_limits<float>::quiet_NaN();
	}

	inline void set(uint32_t index, float value) noexcept {
		if (index < size()) {
			host_data()[index] = value;
		}
	}

};

}
