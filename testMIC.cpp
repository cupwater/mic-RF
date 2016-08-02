#include <stdio.h>
#include <immintrin.h>


#pragma offload_attribute(push, target(mic))

unsigned int random_num = 0;
float  fdata[16] = 
{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
  9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f };

float  f1data[16] = {0.0f};
//{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  //0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

double  ddata[8] = 
{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };


#pragma offload_attribute(pop)

void sample05()
{
    #pragma offload target(mic)
    {
        __m512 v1, v2, v3, v2_1, v3_1;

        v2 = _mm512_setzero_ps();
        v3  = _mm512_setzero_ps();

        v2 = _mm512_loadunpacklo_ps ( v2_1, fdata );
        _mm512_packstorelo_ps ( (void*) (&f1data[0]) , v2 );
        _mm512_packstorehi_ps ( (void*) (&f1data[i]),  v2 );
    }
}

int main(int argc, char *argv[])
{
	sample05();
	int j=0;
    for( j = 0; j < 16; j++)
	{

       printf("  %f\n", f1data[j]);
	}
	return 0;
}
