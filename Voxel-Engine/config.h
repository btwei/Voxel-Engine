#pragma once

enum mode {
	DEFAULT,
	NOCLIP,
	UNMANAGED
};

/* playerControl mode to manage collisions and camera controls */
enum mode playerController = NOCLIP;

/* If true, then uninitialized chunks are initialized according to the generateChunk function */
bool generateChunks = true;

