#include <stdint.h>
#include <stdio.h>
#include <Windows.h>
#include <time.h>
#include <assert.h>

static uint16_t aes_field = 283;
__device__ static uint16_t aes_field_gpu = 283;
static uint8_t* sbox;
// __device__ __constant__ static uint8_t* sbox_gpu;
// static uint8_t inv_sbox[0x100];
__device__ static uint8_t mix_columns_matrix[4][4] = {{2, 1, 1, 3}, {3, 2, 1, 1}, {1, 3, 2, 1}, {1, 1, 3, 2}};
// static uint8_t mix_columns_matrix[4][4] = {{2, 1, 1, 3}, {3, 2, 1, 1}, {1, 3, 2, 1}, {1, 1, 3, 2}};
// static uint8_t inv_mix_columns[4][4] = {{14, 9, 13, 11}, {11, 14, 9, 13}, {13, 11, 14, 9}, {9, 13, 11, 14}};
uint8_t round_constant[4] = {1, 0, 0, 0};


struct aes_encoder {
    uint8_t key_size;
    uint8_t* expanded_key;
};

inline cudaError_t checkCuda(cudaError_t result, int line)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error at line %d: %s\n", line, cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

void rotate_left(uint8_t* bytes) {
    uint8_t first_byte = bytes[1];
    bytes[1] = bytes[2];
    bytes[2] = bytes[3];
    bytes[3] = bytes[0];
    bytes[0] = first_byte;
}

__device__ void sub_bytes_gpu(uint8_t* bytes, int len_bytes, uint8_t* sbox_gpu) {
    for (int i = 0; i < len_bytes; i++) {
        bytes[i] = sbox_gpu[bytes[i]];
    }
}

void sub_bytes(uint8_t* bytes, int len_bytes) {
    for (int i = 0; i < len_bytes; i++) {
        bytes[i] = sbox[bytes[i]];
    }
}

void add_round_constant(uint8_t* bytes) {
    bytes[0] ^= round_constant[0];
    bytes[1] ^= round_constant[1];
    bytes[2] ^= round_constant[2];
    bytes[3] ^= round_constant[3];
    uint16_t new_constant = (uint16_t)round_constant[0] << 1;
    if (new_constant >= 0x100) {
        new_constant ^= aes_field;
    }
    round_constant[0] = (uint8_t)new_constant;
}

// Currently assuming small key.
void key_expansion(struct aes_encoder* aes, uint8_t* initial_key, int key_size) {
    checkCuda(cudaMallocManaged(&(aes->expanded_key), 176 * sizeof(uint8_t)), 66);
    // checkCuda(cudaMemcpy(aes->expanded_key, initial_key, 4 * sizeof(int), cudaMemcpyHostToDevice), 67);
    for (int i = 0; i < key_size; i++) {
        // printf("Copy byte: %d\n", i);
        aes->expanded_key[i] = initial_key[i];
    }
    int round = 0;
    while (round < 10) {
        // printf("Round: %d\n", round);
        int current_key_size = 16 * (round + 1);
        for (int k = 0; k < 4; k++) {
            int i = current_key_size - 4 + (k * 4);
            uint8_t last_four_bytes[] = {aes->expanded_key[i], aes->expanded_key[i + 1], aes->expanded_key[i + 2], aes->expanded_key[i + 3]};
            if (k == 0) {
                rotate_left(last_four_bytes);
                sub_bytes(last_four_bytes, 4);
                add_round_constant(last_four_bytes);
            }
            i = current_key_size - 16 + (k * 4);
            uint8_t first_four_of_last_sixteen_bytes[] = {aes->expanded_key[i], aes->expanded_key[i + 1], aes->expanded_key[i + 2], aes->expanded_key[i + 3]};
            aes->expanded_key[current_key_size + (4 * k)] = first_four_of_last_sixteen_bytes[0] ^ last_four_bytes[0];
            aes->expanded_key[current_key_size + (4 * k) + 1] = first_four_of_last_sixteen_bytes[1] ^ last_four_bytes[1];
            aes->expanded_key[current_key_size + (4 * k) + 2] = first_four_of_last_sixteen_bytes[2] ^ last_four_bytes[2];
            aes->expanded_key[current_key_size + (4 * k) + 3] = first_four_of_last_sixteen_bytes[3] ^ last_four_bytes[3];
        }
        round++;
    }
}

struct aes_encoder* aes_init(int key_size, int* initial_key) {
    struct aes_encoder* e; 
    checkCuda(cudaMallocManaged(&e, sizeof(struct aes_encoder)), 92);
    return e;
}

__device__ void shift_rows(uint8_t* bytes) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < i; j++) {
            uint8_t temp = bytes[0 + i];
            bytes[0 + i] = bytes[4 + i];
            bytes[4 + i] = bytes[8 + i];
            bytes[8 + i] = bytes[12 + i];
            bytes[12 + i] = temp;
        }
    }
}

__device__ uint8_t product(uint16_t a, uint16_t b) {
    uint8_t ret = 0;
    for (int i = 0; i < 8; i++) {
        if ((b & 1) == 1) {
            ret ^= a;
        }
        if ((a & 0x80) == 0x80) {
            a <<= 1;
            a ^= aes_field_gpu;
        }
        else {
            a <<= 1;
        }
        b >>= 1;
    }
    return ret;
}

__device__ void mix_columns(uint8_t* bytes) {
    uint8_t* storage = (uint8_t*)malloc(16 * sizeof(uint8_t));
    // cudaMallocManaged(&storage, 16 * sizeof(uint8_t));
    for (int i = 0; i < 16; i++) {
        storage[i] = 0;
    }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            storage[j + (4 * i)] = product(mix_columns_matrix[0][j], bytes[(i * 4)]) ^ 
                product(mix_columns_matrix[1][j], bytes[(i * 4) + 1]) ^ 
                product(mix_columns_matrix[2][j], bytes[(i * 4) + 2]) ^
                product(mix_columns_matrix[3][j], bytes[(i * 4) + 3]);
        }
    }
    for (int i = 0; i < 16; i++) {
        bytes[i] = storage[i];
    }
}

__device__ void add_round_key(uint8_t* bytes, uint8_t* key_section) {
    for (int i = 0; i < 16; i++) {
        // printf("%d\n", threadIdx.x);
        bytes[i] ^= key_section[i];
    }
}

__device__ void encrypt_16(uint8_t* plain_text, struct aes_encoder* aes, uint8_t* sbox) {
        uint8_t* cipher_text = plain_text;

        int round = 0;
        while (round < 10) {
            uint8_t* key_section = aes->expanded_key + (round * 16); // I love pointer arithmetic!
            add_round_key(cipher_text, key_section);
            sub_bytes_gpu(cipher_text, 16, sbox);
            shift_rows(cipher_text);
            if (round != 9) {
                mix_columns(cipher_text);
            }
            else {
                key_section = aes->expanded_key + 160;
                add_round_key(cipher_text, key_section);
            }
            round++;
        }
}

__global__ void encrypt(uint8_t* plain_text, struct aes_encoder* aes, int length, uint8_t* sbox) {
    for (int i = blockIdx.x * 16; i < length; i += (gridDim.x * 16)) {
        encrypt_16(plain_text + i, aes, sbox);
    }
}

void sbox_init() {
    checkCuda(cudaMallocManaged(&sbox, 0x100), 189);
    FILE* sbox_file = fopen("sbox.txt", "r");
    char s[6];
    while ((fgets(s, 7, sbox_file))) {
        char i[3] = {s[0], s[1], 0};
        char sub[3] = {s[3], s[4], 0};
        int i_int = (int)strtol(i, NULL, 16);
        int sub_int = (int)strtol(sub, NULL, 16);
        sbox[i_int] = sub_int;
    }
    fclose(sbox_file);
}

int main(int ac, char** av) {
    sbox_init();
    // checkCuda(cudaMallocManaged(&sbox_gpu, 0x100), 190);
    // checkCuda(cudaMemcpyToSymbol(sbox_gpu, sbox, 0x100), 201);
    // cudaDeviceSynchronize();
    struct aes_encoder* aes = aes_init(0, 0);
    const TCHAR* key_file = TEXT("aes_key");
    HANDLE h_file = CreateFile(key_file, GENERIC_READ, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
    HANDLE h_map = CreateFileMapping(h_file, NULL, PAGE_READONLY, 0, 0, NULL);
    const TCHAR* pt_file = TEXT("plain_text_long");
    HANDLE pt_h_file = CreateFile(pt_file, GENERIC_ALL, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
    if (pt_h_file == INVALID_HANDLE_VALUE) {
        printf("Problem opening file \n");
    }
    HANDLE pt_h_map = CreateFileMapping(pt_h_file, NULL, PAGE_READWRITE, 0, 0, NULL);
    if (pt_h_map == NULL) {
        printf("Problem creating mapping of file \n");
    }
    uint8_t* initial_key = (uint8_t*)MapViewOfFile(h_map, FILE_MAP_READ, 0, 0, 0);
    cudaMallocManaged(&initial_key, 16 * sizeof(uint8_t));

    for (int i = 0; i < 16; i++) {
        initial_key[i] = i + (0x10 * i);
        printf("%02X", initial_key[i]);
    }
    printf("\n");
    uint8_t* base_pt = (uint8_t*)MapViewOfFile(pt_h_map, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, 0);
    uint8_t* test;
    int length = 16000;
    checkCuda(cudaMallocManaged(&test, length), 211);
    checkCuda(cudaMemcpy(test, base_pt, length, cudaMemcpyHostToDevice), 212);
    if (test == NULL) {
        printf("Problem mapping view of file \n");
    }
    // printf("Reached key expansion\n");
    key_expansion(aes, initial_key, 16);
    int device_id;
    int number_of_SMs;
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&number_of_SMs, cudaDevAttrMultiProcessorCount, device_id);
    // How many threads do we need?
    // We've initialized the length to be 16000 (not the best practice, but just for the sake of this project to measure performance boosts.)
    // Each plaintext block is 16 bytes. So 1000 threads should be able to parallelize perfectly fine.
    // To be safe, I'll implement a stepping as well for each thread.
    // printf("Reached encrypt\n");
    int num_blocks = 32 * number_of_SMs;
    encrypt<<<num_blocks, 1>>>(test, aes, length, sbox);
    // printf("Passed encrypt\n");
    checkCuda(cudaDeviceSynchronize(), 227);
    // printf("Checking first error\n");
    checkCuda(cudaGetLastError(), 226);
    // printf("Printing block\n");
    for (int i = 0; i < 16; i++) {
        printf("%02X", test[i]);
    }
    // printf("Closing handles\n");
    CloseHandle(h_map);
    CloseHandle(h_file);
    CloseHandle(pt_h_map);
    CloseHandle(pt_h_file);
    return 0;
}