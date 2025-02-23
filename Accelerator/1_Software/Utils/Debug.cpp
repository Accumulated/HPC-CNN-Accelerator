#include "CommonInclude.h"
#include "Debug.h"


void show_me_enhanced_from_devince(Matrix *ptr, const char *notification)
{
    // Matrix H_OUT;
/*
    set_allocate_Host(&H_OUT, ptr -> height, ptr -> width, ptr -> depth);

    just_copy_DTH(&H_OUT, ptr, "show_device_elements");

    show_out = 1;
    show_me_enhanced(&H_OUT, notification);
    show_out = 0;*/
}


void show_me_enhanced(Matrix* ptr, const char* NamePtr)
{
/*
    if(show_out == 1)
    {
      setvbuf(stdout, NULL, _IOLBF, 0);

          printf("%s,"
              "it has height = %d, "
              "width = %d, "
              "depth = %d \n",
              NamePtr, ptr->height, ptr->width, ptr->depth);

          printf("{\n");
          for (int i = 0; i < ptr -> height * ptr -> width * ptr -> depth; i++)
          {
              if (i % ptr->width == 0 && i >= ptr->width)
                  printf("\n");

              if (i % (ptr->width * ptr->height) == 0 && i >= (ptr->width * ptr->height));
                  //printf("\n");

              printf("%.8f", ptr->elements[i]);
              if (i + 1 == ptr->height * ptr->width * ptr->depth);
              else
                  printf(", ");
          }

          printf("} \n");
          printf("\n");

          setvbuf(stdout, NULL, _IOLBF, 0);
    }*/
}
