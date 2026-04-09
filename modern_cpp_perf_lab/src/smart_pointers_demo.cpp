#include "lab/common.hpp"

#include <iostream>
#include <memory>
#include <string>
#include <utility>

namespace {

struct Texture {
  explicit Texture(std::string name) : name(std::move(name)) {
    std::cout << "Texture acquired: " << this->name << "\n";
  }

  ~Texture() { std::cout << "Texture released: " << name << "\n"; }

  std::string name;
};

struct Node {
  explicit Node(std::string name) : name(std::move(name)) {
    std::cout << "Node created: " << this->name << "\n";
  }

  ~Node() { std::cout << "Node destroyed: " << name << "\n"; }

  std::string name;
  std::shared_ptr<Node> child;
  std::weak_ptr<Node> parent;
};

void unique_ptr_demo() {
  lab::print_divider("std::unique_ptr");

  std::unique_ptr<Texture> texture = std::make_unique<Texture>("diffuse-map");
  std::cout << "texture owns: " << texture->name << "\n";

  std::unique_ptr<Texture> next_owner = std::move(texture);
  std::cout << "after move, texture == nullptr ? " << std::boolalpha << (texture == nullptr)
            << "\n";
  std::cout << "next_owner now owns: " << next_owner->name << "\n";

  Texture* raw_view = next_owner.get();
  std::cout << "raw_view only observes: " << raw_view->name << "\n";
}

void shared_ptr_demo() {
  lab::print_divider("std::shared_ptr");

  auto model = std::make_shared<Texture>("character-mesh");
  std::cout << "initial use_count=" << model.use_count() << "\n";

  {
    std::shared_ptr<Texture> renderer = model;
    std::cout << "after renderer copy, use_count=" << model.use_count() << "\n";

    {
      std::shared_ptr<Texture> cache = renderer;
      std::cout << "after cache copy, use_count=" << model.use_count() << "\n";
      std::cout << "shared object name=" << cache->name << "\n";
    }

    std::cout << "after cache scope, use_count=" << model.use_count() << "\n";
  }

  std::cout << "after renderer scope, use_count=" << model.use_count() << "\n";
}

void weak_ptr_demo() {
  lab::print_divider("std::weak_ptr");

  auto root = std::make_shared<Node>("root");
  auto leaf = std::make_shared<Node>("leaf");

  root->child = leaf;
  leaf->parent = root;

  std::cout << "root use_count=" << root.use_count() << ", leaf use_count=" << leaf.use_count()
            << "\n";

  if (auto parent = leaf->parent.lock()) {
    std::cout << "leaf can temporarily access parent: " << parent->name << "\n";
  }

  std::weak_ptr<Node> observer = root;
  root.reset();

  std::cout << "after root.reset(), observer.expired()=" << observer.expired() << "\n";
  if (auto locked = observer.lock()) {
    std::cout << "observer still sees: " << locked->name << "\n";
  } else {
    std::cout << "observer.lock() returned empty because the object was destroyed\n";
  }
}

}  // namespace

int main() {
  unique_ptr_demo();
  shared_ptr_demo();
  weak_ptr_demo();

  std::cout << "\nSummary:\n"
               "1. unique_ptr means exclusive ownership and is move-only.\n"
               "2. shared_ptr means shared ownership with reference counting.\n"
               "3. weak_ptr observes a shared object without extending its lifetime.\n";
  return 0;
}
