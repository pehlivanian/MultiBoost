#ifndef __ANALYTIC_UTILS_HPP__
#define __ANALYTIC_UTILS_HPP__

#include <mlpack/core.hpp>

using namespace arma;

namespace ANALYTIC_utils {

  template<bool B>
  using bool_constant = std::integral_constant<bool, B>;

  template<typename DataType>
  struct _is_double : bool_constant<std::is_same<double, typename std::remove_cv<DataType>::type>::value> {};

  using arma_dvec_type = rowvec;
  using arma_fvec_type = frowvec;
  using arma_dmat_type = dmat;
  using arma_fmat_type = fmat;

  template<typename DataType>
  struct arma_vec_type {
    typedef arma_fvec_type value;
  };

  template<>
  struct arma_vec_type<double> {
    typedef arma_dvec_type value;
  };

  template<typename DataType>
  struct arma_mat_type {
    typedef arma_fmat_type value;
  };

  template<>
  struct arma_mat_type<double> {
    typedef arma_dmat_type value;
  };

  template<typename DataType>
  struct arma_types {
    typedef typename arma_vec_type<DataType>::value vectype;
    typedef typename arma_mat_type<DataType>::value mattype;
  };

  // Zip functionality
  template<typename T>
  using select_iterator_for = std::conditional_t<
    std::is_const_v<std::remove_reference_t<T>>,
    typename std::decay_t<T>::const_iterator,
    typename std::decay_t<T>::iterator>;

  template<typename Iter>
  using select_access_type_for = std::conditional_t<
    std::is_same_v<Iter, std::vector<bool>::iterator> ||
    std::is_same_v<Iter, std::vector<bool>::const_iterator>,
    typename std::iterator_traits<Iter>::value_type,
    typename std::iterator_traits<Iter>::reference
    >;

  template<typename Iter1, typename Iter2>
  class zip_iterator {
  public:
    using value_type = std::pair<
    select_access_type_for<Iter1>,
    select_access_type_for<Iter2>
    >;
    using pointer_type = std::pair<
      typename std::iterator_traits<Iter1>::pointer,
      typename std::iterator_traits<Iter2>::pointer
    >;
  
    zip_iterator() = delete;
  
    zip_iterator(Iter1 iter_1_begin, Iter2 iter_2_begin) :
      m_iter_1_begin{iter_1_begin}, 
      m_iter_2_begin{iter_2_begin} {}

    // Need:
    // auto operator++() -> zip_iterator&
    // auto operator++(int) -> zip_iterator
    // auto operator!=(zip_iterator const& rhs)
    // auto operator*() -> value_type

    // From __gnu_cxx::__normal_iterator : 
    // https://gcc.gnu.org/onlinedocs/libstdc++/libstdc++-html-USERS-4.2/class____gnu__cxx_1_1____normal__iterator.html
    /*
      Public Member Functions
      template<typename _Iter> __normal_iterator (
      const __normal_iterator< _Iter, typename __enable_if< (std::__are_same< _Iter, 
      typename _Container::pointer >::__value), 
      _Container >::__type > &__i
      )
      __normal_iterator (const _Iterator &__i)
      __normal_iterator ()
      const _Iterator & base () const
      reference operator * () const
      __normal_iterator operator+ (const difference_type &__n) const
      __normal_iterator operator++ (int)
      __normal_iterator & operator++ ()
      __normal_iterator & operator+= (const difference_type &__n)
      __normal_iterator operator- (const difference_type &__n) const
      __normal_iterator operator-- (int)
      __normal_iterator & operator-- ()
      __normal_iterator & operator-= (const difference_type &__n)
      pointer operator-> () const
      reference operator[] (const difference_type &__n) const
    */

    zip_iterator(const zip_iterator& rhs) : 
      m_iter_1_begin{rhs.m_iter_1_begin},
      m_iter_2_begin{rhs.m_iter_2_begin} {}

    auto operator+(int n) -> zip_iterator {
      (*this)+=n;
      return *this;
    }

    auto operator-(int n) -> zip_iterator {
      (*this)-=n;
      return *this;
    }

    auto operator++() -> zip_iterator& {
      ++m_iter_1_begin;
      ++m_iter_2_begin;
      return *this;
    }

    auto operator++(int) -> zip_iterator {
      auto tmp = *this;
      ++*this;
      return tmp;
    }

    auto operator--() -> zip_iterator& {
      --m_iter_1_begin;
      --m_iter_2_begin;
      return *this;
    }

    auto operator--(int) -> zip_iterator {
      auto tmp = *this;
      --*this;
      return tmp;
    }
  
    auto operator!=(zip_iterator const& rhs) {
      return !(*this == rhs);
    }

    auto operator==(zip_iterator const& rhs) {
      return rhs.m_iter_1_begin == m_iter_1_begin ||
	rhs.m_iter_2_begin == m_iter_2_begin;
    }

    auto operator*() -> value_type {
      return value_type{*m_iter_1_begin, *m_iter_2_begin};
    }

    auto operator->() -> pointer_type {
      return pointer_type{m_iter_1_begin, m_iter_2_begin};
    }

    auto operator=(zip_iterator const& rhs) -> zip_iterator& {
      m_iter_1_begin = rhs.m_iter_1_begin;
      m_iter_2_begin = rhs.m_iter_2_begin;
      return *this;
    }

    auto operator=(zip_iterator &&rhs) -> zip_iterator& {
      m_iter_1_begin = std::move(rhs.m_iter_1_begin);
      m_iter_2_begin = std::move(rhs.m_iter_2_begin);
      return *this;
    }

    auto operator+=(int n) -> zip_iterator& {
      m_iter_1_begin+=n;
      m_iter_2_begin+=n;
      return *this;
    }
  

  private:
    Iter1 m_iter_1_begin;
    Iter2 m_iter_2_begin;
  };


  template<typename T, typename U>
  class zipper {
  public:
    using Iter1 = select_iterator_for<T>;
    using Iter2 = select_iterator_for<U>;
    using zip_type = zip_iterator<Iter1, Iter2>;

    template<typename V, typename W>
    explicit zipper(V &&a, W &&b) :
      m_a{std::forward<V>(a)},
      m_b{std::forward<W>(b)} {}

    auto begin() -> zip_type {
      return zip_type{std::begin(m_a), std::begin(m_b)};
    }
    auto end() -> zip_type {
      return zip_type{std::end(m_a), std::end(m_b)};
    }
  private:
    T m_a;
    U m_b;

  };

  template<typename T, typename U>
  auto zip(T &&t, U &&u) {
    return zipper<T, U>{std::forward<T>(t), std::forward<U>(u)};
  }

} // namespace ANALYTIC_utils

#endif
