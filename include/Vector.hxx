#pragma once

#include <cstdint>
#include <iostream>
#include <memory>

class Vector {
	mutable enum class State : uint8_t {
		SYNCED, ON_HOST, ON_DEVICE
	} _state;

	uint32_t _size, _totalSize;
	mutable std::shared_ptr<float[]> _hostBuffer;
	mutable std::shared_ptr<float[]> _deviceData;

public:
	// Constructors 

	Vector(uint32_t size);
	Vector(uint32_t size, float fillValue);
	Vector(const Vector &vector);
	Vector(Vector &vector) = default;

	// Destructor

	virtual ~Vector() = default;

	// Accessors

	inline uint32_t size() const { return _size; }
	inline uint32_t total_size() const { return _totalSize; }

	// Mutators

	void fill(float value);
	void add(const Vector &vector);
	void subtract(const Vector &vector);

	// Operators

	Vector &operator=(float value);

	Vector &operator+=(const Vector &target);
	Vector operator+(const Vector &target) const;

	Vector &operator-=(const Vector &target);
	Vector operator-(const Vector &target) const;
	
	float operator[](uint32_t index) const;
	friend std::ostream &operator<<(std::ostream &stream, const Vector &vector);

private:
	static void _size_test(uint32_t left, uint32_t right);

	void _toHost() const;
	void _toDevice() const;
};
