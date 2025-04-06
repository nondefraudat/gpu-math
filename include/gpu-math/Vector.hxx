#pragma once

#include "types.hxx"
#include <iostream>

namespace gm {

class Vector : public Shared {
public:
	// Constructors
	Vector(size_t size, real_t fillValue = static_cast<real_t>(0)) 
		: Shared(size, fillValue) { }

	// Mutators

	void add(const Vector &vector);
	void subtract(const Vector &vector);

	// Operators

	Vector &operator=(real_t value);

	Vector &operator+=(const Vector &target);
	Vector operator+(const Vector &target) const;

	Vector &operator-=(const Vector &target);
	Vector operator-(const Vector &target) const;
	
	real_t operator[](size_t index) const;
	friend std::ostream &operator<<(std::ostream &stream, const Vector &vector);
};

}
