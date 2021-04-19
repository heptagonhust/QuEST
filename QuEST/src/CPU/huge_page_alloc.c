#include <sys/mman.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include "huge_page_alloc.h"
#define PROTECTION (PROT_READ | PROT_WRITE)
#define ADDR (void *)(0x0UL)
#define FLAGS (MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB)

void *huge_alloc(size_t len) {
  void *addr = mmap(ADDR, len, PROTECTION, FLAGS, -1, 0);
  if (addr == MAP_FAILED) {
    perror("mmap failed: ");
    exit(127);
  }
  return addr;
}

void huge_free(void *addr) {
  // do nothing
  return;
}
