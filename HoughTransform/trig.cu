﻿#include "trig.cuh"
//#define _USE_MATH_DEFINES
//#include <math.h>
//// Precompute the sin and cos values. Print them in hex.
//union float_bits {
//	int i;
//	float f;
//};
//union float_bits bits;
//for (int i = 0; i < 360; i++) {
//	//bits.f = sin((i * M_PI) / 180.0);
//	bits.f = cos((i * M_PI) / 180.0);
//	cout << "0x" << hex << bits.i << ",\n";
//}

const unsigned int sin_bits[NUM_DEGREE_CALC] = {
	0x0,
	0x3c8ef859,
	0x3d0ef2c6,
	0x3d565e3a,
	0x3d8edc7b,
	0x3db27eb6,
	0x3dd61305,
	0x3df996a2,
	0x3e0e8365,
	0x3e20305b,
	0x3e31d0d4,
	0x3e43636f,
	0x3e54e6cd,
	0x3e665992,
	0x3e77ba60,
	0x3e8483ee,
	0x3e8d2057,
	0x3e95b1be,
	0x3e9e377a,
	0x3ea6b0df,
	0x3eaf1d44,
	0x3eb77c01,
	0x3ebfcc6f,
	0x3ec80de9,
	0x3ed03fc9,
	0x3ed8616c,
	0x3ee0722f,
	0x3ee87171,
	0x3ef05e94,
	0x3ef838f7,
	0x3f000000,
	0x3f03d989,
	0x3f07a8ca,
	0x3f0b6d77,
	0x3f0f2744,
	0x3f12d5e8,
	0x3f167918,
	0x3f1a108d,
	0x3f1d9bfe,
	0x3f211b24,
	0x3f248dbb,
	0x3f27f37c,
	0x3f2b4c25,
	0x3f2e9772,
	0x3f31d522,
	0x3f3504f3,
	0x3f3826a7,
	0x3f3b39ff,
	0x3f3e3ebd,
	0x3f4134a6,
	0x3f441b7d,
	0x3f46f30a,
	0x3f49bb13,
	0x3f4c7360,
	0x3f4f1bbd,
	0x3f51b3f3,
	0x3f543bce,
	0x3f56b31d,
	0x3f5919ae,
	0x3f5b6f51,
	0x3f5db3d7,
	0x3f5fe714,
	0x3f6208da,
	0x3f641901,
	0x3f66175e,
	0x3f6803ca,
	0x3f69de1d,
	0x3f6ba635,
	0x3f6d5bec,
	0x3f6eff20,
	0x3f708fb2,
	0x3f720d81,
	0x3f737871,
	0x3f74d063,
	0x3f76153f,
	0x3f7746ea,
	0x3f78654d,
	0x3f797051,
	0x3f7a67e2,
	0x3f7b4beb,
	0x3f7c1c5c,
	0x3f7cd925,
	0x3f7d8235,
	0x3f7e1781,
	0x3f7e98fd,
	0x3f7f069e,
	0x3f7f605c,
	0x3f7fa62f,
	0x3f7fd814,
	0x3f7ff605,
	0x3f800000,
	0x3f7ff605,
	0x3f7fd814,
	0x3f7fa62f,
	0x3f7f605c,
	0x3f7f069e,
	0x3f7e98fd,
	0x3f7e1781,
	0x3f7d8235,
	0x3f7cd925,
	0x3f7c1c5c,
	0x3f7b4beb,
	0x3f7a67e2,
	0x3f797051,
	0x3f78654d,
	0x3f7746ea,
	0x3f76153f,
	0x3f74d063,
	0x3f737871,
	0x3f720d81,
	0x3f708fb2,
	0x3f6eff20,
	0x3f6d5bec,
	0x3f6ba635,
	0x3f69de1d,
	0x3f6803ca,
	0x3f66175e,
	0x3f641901,
	0x3f6208da,
	0x3f5fe714,
	0x3f5db3d7,
	0x3f5b6f51,
	0x3f5919ae,
	0x3f56b31d,
	0x3f543bce,
	0x3f51b3f3,
	0x3f4f1bbd,
	0x3f4c7360,
	0x3f49bb13,
	0x3f46f30a,
	0x3f441b7d,
	0x3f4134a6,
	0x3f3e3ebd,
	0x3f3b39ff,
	0x3f3826a7,
	0x3f3504f3,
	0x3f31d522,
	0x3f2e9772,
	0x3f2b4c25,
	0x3f27f37c,
	0x3f248dbb,
	0x3f211b24,
	0x3f1d9bfe,
	0x3f1a108d,
	0x3f167918,
	0x3f12d5e8,
	0x3f0f2744,
	0x3f0b6d77,
	0x3f07a8ca,
	0x3f03d989,
	0x3f000000,
	0x3ef838f7,
	0x3ef05e94,
	0x3ee87171,
	0x3ee0722f,
	0x3ed8616c,
	0x3ed03fc9,
	0x3ec80de9,
	0x3ebfcc6f,
	0x3eb77c01,
	0x3eaf1d44,
	0x3ea6b0df,
	0x3e9e377a,
	0x3e95b1be,
	0x3e8d2057,
	0x3e8483ee,
	0x3e77ba60,
	0x3e665992,
	0x3e54e6cd,
	0x3e43636f,
	0x3e31d0d4,
	0x3e20305b,
	0x3e0e8365,
	0x3df996a2,
	0x3dd61305,
	0x3db27eb6,
	0x3d8edc7b,
	0x3d565e3a,
	0x3d0ef2c6,
	0x3c8ef859,
	0x250d3132,
	0xbc8ef859,
	0xbd0ef2c6,
	0xbd565e3a,
	0xbd8edc7b,
	0xbdb27eb6,
	0xbdd61305,
	0xbdf996a2,
	0xbe0e8365,
	0xbe20305b,
	0xbe31d0d4,
	0xbe43636f,
	0xbe54e6cd,
	0xbe665992,
	0xbe77ba60,
	0xbe8483ee,
	0xbe8d2057,
	0xbe95b1be,
	0xbe9e377a,
	0xbea6b0df,
	0xbeaf1d44,
	0xbeb77c01,
	0xbebfcc6f,
	0xbec80de9,
	0xbed03fc9,
	0xbed8616c,
	0xbee0722f,
	0xbee87171,
	0xbef05e94,
	0xbef838f7,
	0xbf000000,
	0xbf03d989,
	0xbf07a8ca,
	0xbf0b6d77,
	0xbf0f2744,
	0xbf12d5e8,
	0xbf167918,
	0xbf1a108d,
	0xbf1d9bfe,
	0xbf211b24,
	0xbf248dbb,
	0xbf27f37c,
	0xbf2b4c25,
	0xbf2e9772,
	0xbf31d522,
	0xbf3504f3,
	0xbf3826a7,
	0xbf3b39ff,
	0xbf3e3ebd,
	0xbf4134a6,
	0xbf441b7d,
	0xbf46f30a,
	0xbf49bb13,
	0xbf4c7360,
	0xbf4f1bbd,
	0xbf51b3f3,
	0xbf543bce,
	0xbf56b31d,
	0xbf5919ae,
	0xbf5b6f51,
	0xbf5db3d7,
	0xbf5fe714,
	0xbf6208da,
	0xbf641901,
	0xbf66175e,
	0xbf6803ca,
	0xbf69de1d,
	0xbf6ba635,
	0xbf6d5bec,
	0xbf6eff20,
	0xbf708fb2,
	0xbf720d81,
	0xbf737871,
	0xbf74d063,
	0xbf76153f,
	0xbf7746ea,
	0xbf78654d,
	0xbf797051,
	0xbf7a67e2,
	0xbf7b4beb,
	0xbf7c1c5c,
	0xbf7cd925,
	0xbf7d8235,
	0xbf7e1781,
	0xbf7e98fd,
	0xbf7f069e,
	0xbf7f605c,
	0xbf7fa62f,
	0xbf7fd814,
	0xbf7ff605,
	0xbf800000,
	0xbf7ff605,
	0xbf7fd814,
	0xbf7fa62f,
	0xbf7f605c,
	0xbf7f069e,
	0xbf7e98fd,
	0xbf7e1781,
	0xbf7d8235,
	0xbf7cd925,
	0xbf7c1c5c,
	0xbf7b4beb,
	0xbf7a67e2,
	0xbf797051,
	0xbf78654d,
	0xbf7746ea,
	0xbf76153f,
	0xbf74d063,
	0xbf737871,
	0xbf720d81,
	0xbf708fb2,
	0xbf6eff20,
	0xbf6d5bec,
	0xbf6ba635,
	0xbf69de1d,
	0xbf6803ca,
	0xbf66175e,
	0xbf641901,
	0xbf6208da,
	0xbf5fe714,
	0xbf5db3d7,
	0xbf5b6f51,
	0xbf5919ae,
	0xbf56b31d,
	0xbf543bce,
	0xbf51b3f3,
	0xbf4f1bbd,
	0xbf4c7360,
	0xbf49bb13,
	0xbf46f30a,
	0xbf441b7d,
	0xbf4134a6,
	0xbf3e3ebd,
	0xbf3b39ff,
	0xbf3826a7,
	0xbf3504f3,
	0xbf31d522,
	0xbf2e9772,
	0xbf2b4c25,
	0xbf27f37c,
	0xbf248dbb,
	0xbf211b24,
	0xbf1d9bfe,
	0xbf1a108d,
	0xbf167918,
	0xbf12d5e8,
	0xbf0f2744,
	0xbf0b6d77,
	0xbf07a8ca,
	0xbf03d989,
	0xbf000000,
	0xbef838f7,
	0xbef05e94,
	0xbee87171,
	0xbee0722f,
	0xbed8616c,
	0xbed03fc9,
	0xbec80de9,
	0xbebfcc6f,
	0xbeb77c01,
	0xbeaf1d44,
	0xbea6b0df,
	0xbe9e377a,
	0xbe95b1be,
	0xbe8d2057,
	0xbe8483ee,
	0xbe77ba60,
	0xbe665992,
	0xbe54e6cd,
	0xbe43636f,
	0xbe31d0d4,
	0xbe20305b,
	0xbe0e8365,
	0xbdf996a2,
	0xbdd61305,
	0xbdb27eb6,
	0xbd8edc7b,
	0xbd565e3a,
	0xbd0ef2c6,
	0xbc8ef859
};
const unsigned int cos_bits[NUM_DEGREE_CALC] = {
	0x3f800000,
	0x3f7ff605,
	0x3f7fd814,
	0x3f7fa62f,
	0x3f7f605c,
	0x3f7f069e,
	0x3f7e98fd,
	0x3f7e1781,
	0x3f7d8235,
	0x3f7cd925,
	0x3f7c1c5c,
	0x3f7b4beb,
	0x3f7a67e2,
	0x3f797051,
	0x3f78654d,
	0x3f7746ea,
	0x3f76153f,
	0x3f74d063,
	0x3f737871,
	0x3f720d81,
	0x3f708fb2,
	0x3f6eff20,
	0x3f6d5bec,
	0x3f6ba635,
	0x3f69de1d,
	0x3f6803ca,
	0x3f66175e,
	0x3f641901,
	0x3f6208da,
	0x3f5fe714,
	0x3f5db3d7,
	0x3f5b6f51,
	0x3f5919ae,
	0x3f56b31d,
	0x3f543bce,
	0x3f51b3f3,
	0x3f4f1bbd,
	0x3f4c7360,
	0x3f49bb13,
	0x3f46f30a,
	0x3f441b7d,
	0x3f4134a6,
	0x3f3e3ebd,
	0x3f3b39ff,
	0x3f3826a7,
	0x3f3504f3,
	0x3f31d522,
	0x3f2e9772,
	0x3f2b4c25,
	0x3f27f37c,
	0x3f248dbb,
	0x3f211b24,
	0x3f1d9bfe,
	0x3f1a108d,
	0x3f167918,
	0x3f12d5e8,
	0x3f0f2744,
	0x3f0b6d77,
	0x3f07a8ca,
	0x3f03d989,
	0x3f000000,
	0x3ef838f7,
	0x3ef05e94,
	0x3ee87171,
	0x3ee0722f,
	0x3ed8616c,
	0x3ed03fc9,
	0x3ec80de9,
	0x3ebfcc6f,
	0x3eb77c01,
	0x3eaf1d44,
	0x3ea6b0df,
	0x3e9e377a,
	0x3e95b1be,
	0x3e8d2057,
	0x3e8483ee,
	0x3e77ba60,
	0x3e665992,
	0x3e54e6cd,
	0x3e43636f,
	0x3e31d0d4,
	0x3e20305b,
	0x3e0e8365,
	0x3df996a2,
	0x3dd61305,
	0x3db27eb6,
	0x3d8edc7b,
	0x3d565e3a,
	0x3d0ef2c6,
	0x3c8ef859,
	0x248d3132,
	0xbc8ef859,
	0xbd0ef2c6,
	0xbd565e3a,
	0xbd8edc7b,
	0xbdb27eb6,
	0xbdd61305,
	0xbdf996a2,
	0xbe0e8365,
	0xbe20305b,
	0xbe31d0d4,
	0xbe43636f,
	0xbe54e6cd,
	0xbe665992,
	0xbe77ba60,
	0xbe8483ee,
	0xbe8d2057,
	0xbe95b1be,
	0xbe9e377a,
	0xbea6b0df,
	0xbeaf1d44,
	0xbeb77c01,
	0xbebfcc6f,
	0xbec80de9,
	0xbed03fc9,
	0xbed8616c,
	0xbee0722f,
	0xbee87171,
	0xbef05e94,
	0xbef838f7,
	0xbf000000,
	0xbf03d989,
	0xbf07a8ca,
	0xbf0b6d77,
	0xbf0f2744,
	0xbf12d5e8,
	0xbf167918,
	0xbf1a108d,
	0xbf1d9bfe,
	0xbf211b24,
	0xbf248dbb,
	0xbf27f37c,
	0xbf2b4c25,
	0xbf2e9772,
	0xbf31d522,
	0xbf3504f3,
	0xbf3826a7,
	0xbf3b39ff,
	0xbf3e3ebd,
	0xbf4134a6,
	0xbf441b7d,
	0xbf46f30a,
	0xbf49bb13,
	0xbf4c7360,
	0xbf4f1bbd,
	0xbf51b3f3,
	0xbf543bce,
	0xbf56b31d,
	0xbf5919ae,
	0xbf5b6f51,
	0xbf5db3d7,
	0xbf5fe714,
	0xbf6208da,
	0xbf641901,
	0xbf66175e,
	0xbf6803ca,
	0xbf69de1d,
	0xbf6ba635,
	0xbf6d5bec,
	0xbf6eff20,
	0xbf708fb2,
	0xbf720d81,
	0xbf737871,
	0xbf74d063,
	0xbf76153f,
	0xbf7746ea,
	0xbf78654d,
	0xbf797051,
	0xbf7a67e2,
	0xbf7b4beb,
	0xbf7c1c5c,
	0xbf7cd925,
	0xbf7d8235,
	0xbf7e1781,
	0xbf7e98fd,
	0xbf7f069e,
	0xbf7f605c,
	0xbf7fa62f,
	0xbf7fd814,
	0xbf7ff605,
	0xbf800000,
	0xbf7ff605,
	0xbf7fd814,
	0xbf7fa62f,
	0xbf7f605c,
	0xbf7f069e,
	0xbf7e98fd,
	0xbf7e1781,
	0xbf7d8235,
	0xbf7cd925,
	0xbf7c1c5c,
	0xbf7b4beb,
	0xbf7a67e2,
	0xbf797051,
	0xbf78654d,
	0xbf7746ea,
	0xbf76153f,
	0xbf74d063,
	0xbf737871,
	0xbf720d81,
	0xbf708fb2,
	0xbf6eff20,
	0xbf6d5bec,
	0xbf6ba635,
	0xbf69de1d,
	0xbf6803ca,
	0xbf66175e,
	0xbf641901,
	0xbf6208da,
	0xbf5fe714,
	0xbf5db3d7,
	0xbf5b6f51,
	0xbf5919ae,
	0xbf56b31d,
	0xbf543bce,
	0xbf51b3f3,
	0xbf4f1bbd,
	0xbf4c7360,
	0xbf49bb13,
	0xbf46f30a,
	0xbf441b7d,
	0xbf4134a6,
	0xbf3e3ebd,
	0xbf3b39ff,
	0xbf3826a7,
	0xbf3504f3,
	0xbf31d522,
	0xbf2e9772,
	0xbf2b4c25,
	0xbf27f37c,
	0xbf248dbb,
	0xbf211b24,
	0xbf1d9bfe,
	0xbf1a108d,
	0xbf167918,
	0xbf12d5e8,
	0xbf0f2744,
	0xbf0b6d77,
	0xbf07a8ca,
	0xbf03d989,
	0xbf000000,
	0xbef838f7,
	0xbef05e94,
	0xbee87171,
	0xbee0722f,
	0xbed8616c,
	0xbed03fc9,
	0xbec80de9,
	0xbebfcc6f,
	0xbeb77c01,
	0xbeaf1d44,
	0xbea6b0df,
	0xbe9e377a,
	0xbe95b1be,
	0xbe8d2057,
	0xbe8483ee,
	0xbe77ba60,
	0xbe665992,
	0xbe54e6cd,
	0xbe43636f,
	0xbe31d0d4,
	0xbe20305b,
	0xbe0e8365,
	0xbdf996a2,
	0xbdd61305,
	0xbdb27eb6,
	0xbd8edc7b,
	0xbd565e3a,
	0xbd0ef2c6,
	0xbc8ef859,
	0xa553c9ca,
	0x3c8ef859,
	0x3d0ef2c6,
	0x3d565e3a,
	0x3d8edc7b,
	0x3db27eb6,
	0x3dd61305,
	0x3df996a2,
	0x3e0e8365,
	0x3e20305b,
	0x3e31d0d4,
	0x3e43636f,
	0x3e54e6cd,
	0x3e665992,
	0x3e77ba60,
	0x3e8483ee,
	0x3e8d2057,
	0x3e95b1be,
	0x3e9e377a,
	0x3ea6b0df,
	0x3eaf1d44,
	0x3eb77c01,
	0x3ebfcc6f,
	0x3ec80de9,
	0x3ed03fc9,
	0x3ed8616c,
	0x3ee0722f,
	0x3ee87171,
	0x3ef05e94,
	0x3ef838f7,
	0x3f000000,
	0x3f03d989,
	0x3f07a8ca,
	0x3f0b6d77,
	0x3f0f2744,
	0x3f12d5e8,
	0x3f167918,
	0x3f1a108d,
	0x3f1d9bfe,
	0x3f211b24,
	0x3f248dbb,
	0x3f27f37c,
	0x3f2b4c25,
	0x3f2e9772,
	0x3f31d522,
	0x3f3504f3,
	0x3f3826a7,
	0x3f3b39ff,
	0x3f3e3ebd,
	0x3f4134a6,
	0x3f441b7d,
	0x3f46f30a,
	0x3f49bb13,
	0x3f4c7360,
	0x3f4f1bbd,
	0x3f51b3f3,
	0x3f543bce,
	0x3f56b31d,
	0x3f5919ae,
	0x3f5b6f51,
	0x3f5db3d7,
	0x3f5fe714,
	0x3f6208da,
	0x3f641901,
	0x3f66175e,
	0x3f6803ca,
	0x3f69de1d,
	0x3f6ba635,
	0x3f6d5bec,
	0x3f6eff20,
	0x3f708fb2,
	0x3f720d81,
	0x3f737871,
	0x3f74d063,
	0x3f76153f,
	0x3f7746ea,
	0x3f78654d,
	0x3f797051,
	0x3f7a67e2,
	0x3f7b4beb,
	0x3f7c1c5c,
	0x3f7cd925,
	0x3f7d8235,
	0x3f7e1781,
	0x3f7e98fd,
	0x3f7f069e,
	0x3f7f605c,
	0x3f7fa62f,
	0x3f7fd814,
	0x3f7ff605
};