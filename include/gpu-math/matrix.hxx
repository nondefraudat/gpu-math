#pragma once

#include <cstdint>
#include <memory>
#include <limits>
#include <iostream>

namespace gpu_math {

class matrix {
	uint16_t _width, _height, _size, _bytes_count;
	mutable std::shared_ptr<float[]> _host_data;
	mutable std::shared_ptr<float[]> _device_data;
	mutable bool _on_device;

public:
	matrix(uint16_t height, uint16_t width) noexcept;
	matrix(uint16_t height, uint16_t width, float fill_value) noexcept
			: matrix(height, width) { fill(fill_value); }

	inline uint16_t width() const noexcept { return _width; }
	inline uint16_t height() const noexcept { return _height; }
	inline uint16_t size() const noexcept { return _size; }
	inline uint16_t bytes_count() const noexcept { return _bytes_count; }

	inline float get(uint16_t row, uint16_t column) const noexcept {
		return get(index(row, column));
	}

	inline void set(uint16_t row, uint16_t column, float value) noexcept {
		set(index(row, column), value);
	}

	void fill(float value) noexcept;
	matrix multiply(const matrix &other) const noexcept;

	inline matrix operator*(const matrix &other) const noexcept {
		return multiply(other);
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

	inline float *host_data() const noexcept {
		to_host(); return _host_data.get();
	}
	inline float *device_data() const noexcept {
		to_device(); return _device_data.get();
	}

	void to_host() const noexcept;
	void to_device() const noexcept;

};

}
