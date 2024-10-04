#pragma once

enum mode {
	DEFAULT,
	NOCLIP,
	UNMANAGED
};

/* playerControl mode to manage collisions and camera controls */
enum mode playerController = NOCLIP;

/* If true, then uninitialized chunks are initialized according to the generateChunk function */
bool generateChunks = false;

/* If true, then chunks will automatically be loaded and unloaded based on view distance */
/* VIEW_DISTANCE is the number of chunks away from the player that are loaded */
bool useViewDistance = true;
const int VIEW_DISTANCE = 3;

/* TEXTURE ATLAS */
/* All block textures should be padded by at least 1 px of repeated edge to prevent texture bleeding */
const int TEXTURE_DIM = 16;
const int TEXTURE_PADDING = 1;
/* If the blocks use a special top-side-bottom face scheme, include the faces in that order in the atlas and include the first block id in this list */
/* Block ids are 0 indexed! */
const std::set<int> block3faced = {1};

/* Feel free to alias your blockIds through an enum, like so */
enum blockType : short {
	UNDEFINED,
	GRASS,
	DIRT = 3,
	STONE,
	SAND
};