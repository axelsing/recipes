#include <iostream>
#include <memory>

struct B;
struct A {
  ~A() {
    std::cout << "A dtor" << std::endl;
  }
  
  std::shared_ptr<B> b_;
};
struct B {
  ~B() {
    std::cout << "B dtor" << std::endl;
  }
  
  //std::shared_ptr<A> a_;
  std::weak_ptr<A> a_;
};

int main() {
  {
    auto pa = std::make_shared<A>();
    auto pb = std::make_shared<B>();
    std::cout << "pa.use_count:" << pa.use_count() << std::endl;
  }
  {
    auto pa = std::make_shared<A>();
    auto pb = std::make_shared<B>();
    pa->b_ = pb;
    pb->a_ = pa;
    std::cout << "pa.use_count:" << pa.use_count() << std::endl;
  }
  
  return 0;
}
