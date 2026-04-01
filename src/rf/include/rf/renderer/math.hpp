#ifndef RF_MATH_HPP
#define RF_MATH_HPP

#include <cstddef>
#include <type_traits>

namespace rf
{

/// 2D -> 1D mapping
/// for reference see http://szudzik.com/ElegantPairing.pdf
/// input values must be kept small, because unique value grows quadratically

template <class T>
inline void elegant_pair(std::size_t &x, const T &y)
{
    static_assert(std::is_convertible<T, std::size_t>::value,
                  "T type shall be convertible to size_t");
    std::size_t v = static_cast<std::size_t>(y);

    x = (x * x + 3 * x + 2 * x * v + v + v * v) / 2;
}

} // namespace rf

#endif
