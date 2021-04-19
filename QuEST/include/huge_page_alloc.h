#ifndef HUGE_PAGE_ALLOC
#define HUGE_PAGE_ALLOC
#include <inttypes.h>
#include <stdlib.h>
void *huge_alloc(size_t len);
void huge_free(void *addr);
#endif
