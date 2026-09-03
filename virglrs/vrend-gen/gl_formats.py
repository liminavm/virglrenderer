# Converted from src/vrend/vrend_formats.c (the C reference), table by table. A row is
# (virgl format, internalformat, format, type, swizzle, view class).

NO='NO_SWIZZLE'

base_rgba_formats = [
    ('R8G8_R8B8_422_UNORM', 'GL_RGBA8', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_32'),
    ('R8G8B8X8_UNORM', 'GL_RGBA8', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'RGB1_SWIZZLE', 'view_class_32'),
    ('R8G8B8A8_UNORM', 'GL_RGBA8', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_32'),
    ('A8R8G8B8_UNORM', 'GL_RGBA8', 'GL_BGRA', 'GL_UNSIGNED_INT_8_8_8_8', 'NO_SWIZZLE', 'view_class_32'),
    ('X8R8G8B8_UNORM', 'GL_RGBA8', 'GL_BGRA', 'GL_UNSIGNED_INT_8_8_8_8', 'NO_SWIZZLE', 'view_class_32'),
    ('A8B8G8R8_UNORM', 'GL_RGBA8', 'GL_ABGR_EXT', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_32'),
    ('B4G4R4X4_UNORM', 'GL_RGBA4', 'GL_BGRA', 'GL_UNSIGNED_SHORT_4_4_4_4_REV', 'RGB1_SWIZZLE', 'view_class_unsupported'),
    ('A4B4G4R4_UNORM', 'GL_RGBA4', 'GL_RGBA', 'GL_UNSIGNED_SHORT_4_4_4_4', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('B5G5R5X1_UNORM', 'GL_RGB5_A1', 'GL_BGRA', 'GL_UNSIGNED_SHORT_1_5_5_5_REV', 'RGB1_SWIZZLE', 'view_class_unsupported'),
    ('B5G6R5_UNORM', 'GL_RGB565', 'GL_RGB', 'GL_UNSIGNED_SHORT_5_6_5', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('B2G3R3_UNORM', 'GL_R3_G3_B2', 'GL_RGB', 'GL_UNSIGNED_BYTE_3_3_2', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('R16G16B16X16_UNORM', 'GL_RGBA16', 'GL_RGBA', 'GL_UNSIGNED_SHORT', 'RGB1_SWIZZLE', 'view_class_64'),
    ('R16G16B16A16_UNORM', 'GL_RGBA16', 'GL_RGBA', 'GL_UNSIGNED_SHORT', 'NO_SWIZZLE', 'view_class_64'),
]

gl_base_rgba_formats = [
    ('B4G4R4A4_UNORM', 'GL_RGBA4', 'GL_BGRA', 'GL_UNSIGNED_SHORT_4_4_4_4_REV', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('B5G5R5A1_UNORM', 'GL_RGB5_A1', 'GL_BGRA', 'GL_UNSIGNED_SHORT_1_5_5_5_REV', 'NO_SWIZZLE', 'view_class_unsupported'),
]

base_depth_formats = [
    ('Z16_UNORM', 'GL_DEPTH_COMPONENT16', 'GL_DEPTH_COMPONENT', 'GL_UNSIGNED_SHORT', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('S8_UINT_Z24_UNORM', 'GL_DEPTH24_STENCIL8_EXT', 'GL_DEPTH_STENCIL', 'GL_UNSIGNED_INT_24_8', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('Z24X8_UNORM', 'GL_DEPTH_COMPONENT24', 'GL_DEPTH_COMPONENT', 'GL_UNSIGNED_INT', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('Z32_FLOAT', 'GL_DEPTH_COMPONENT32F', 'GL_DEPTH_COMPONENT', 'GL_FLOAT', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('Z32_FLOAT_S8X24_UINT', 'GL_DEPTH32F_STENCIL8', 'GL_DEPTH_STENCIL', 'GL_FLOAT_32_UNSIGNED_INT_24_8_REV', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('X24S8_UINT', 'GL_STENCIL_INDEX8', 'GL_STENCIL_INDEX', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_unsupported'),
]

gl_z32_format = [
    ('Z32_UNORM', 'GL_DEPTH_COMPONENT32', 'GL_DEPTH_COMPONENT', 'GL_UNSIGNED_INT', 'NO_SWIZZLE', 'view_class_unsupported'),
]

gles_z32_format = [
    ('Z32_UNORM', 'GL_DEPTH_COMPONENT24', 'GL_DEPTH_COMPONENT', 'GL_UNSIGNED_INT', 'NO_SWIZZLE', 'view_class_unsupported'),
]

rg_base_formats = [
    ('R8_UNORM', 'GL_R8', 'GL_RED', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_8'),
    ('R8G8_UNORM', 'GL_RG8', 'GL_RG', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_16'),
    ('R16_UNORM', 'GL_R16', 'GL_RED', 'GL_UNSIGNED_SHORT', 'NO_SWIZZLE', 'view_class_16'),
    ('R16G16_UNORM', 'GL_RG16', 'GL_RG', 'GL_UNSIGNED_SHORT', 'NO_SWIZZLE', 'view_class_32'),
]

integer_base_formats = [
    ('R8G8B8A8_UINT', 'GL_RGBA8UI', 'GL_RGBA_INTEGER', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_32'),
    ('R8G8B8A8_SINT', 'GL_RGBA8I', 'GL_RGBA_INTEGER', 'GL_BYTE', 'NO_SWIZZLE', 'view_class_32'),
    ('R16G16B16A16_UINT', 'GL_RGBA16UI', 'GL_RGBA_INTEGER', 'GL_UNSIGNED_SHORT', 'NO_SWIZZLE', 'view_class_32'),
    ('R16G16B16A16_SINT', 'GL_RGBA16I', 'GL_RGBA_INTEGER', 'GL_SHORT', 'NO_SWIZZLE', 'view_class_32'),
    ('R32G32B32A32_UINT', 'GL_RGBA32UI', 'GL_RGBA_INTEGER', 'GL_UNSIGNED_INT', 'NO_SWIZZLE', 'view_class_32'),
    ('R32G32B32A32_SINT', 'GL_RGBA32I', 'GL_RGBA_INTEGER', 'GL_INT', 'NO_SWIZZLE', 'view_class_32'),
]

integer_3comp_formats = [
    ('R8G8B8X8_UINT', 'GL_RGBA8UI', 'GL_RGBA_INTEGER', 'GL_UNSIGNED_BYTE', 'RGB1_SWIZZLE', 'view_class_32'),
    ('R8G8B8X8_SINT', 'GL_RGBA8I', 'GL_RGBA_INTEGER', 'GL_BYTE', 'RGB1_SWIZZLE', 'view_class_32'),
    ('R16G16B16X16_UINT', 'GL_RGBA16UI', 'GL_RGBA_INTEGER', 'GL_UNSIGNED_SHORT', 'RGB1_SWIZZLE', 'view_class_64'),
    ('R16G16B16X16_SINT', 'GL_RGBA16I', 'GL_RGBA_INTEGER', 'GL_SHORT', 'RGB1_SWIZZLE', 'view_class_64'),
    ('R32G32B32_UINT', 'GL_RGB32UI', 'GL_RGB_INTEGER', 'GL_UNSIGNED_INT', 'NO_SWIZZLE', 'view_class_96'),
    ('R32G32B32_SINT', 'GL_RGB32I', 'GL_RGB_INTEGER', 'GL_INT', 'NO_SWIZZLE', 'view_class_96'),
]

float_base_formats = [
    ('R16G16B16A16_FLOAT', 'GL_RGBA16F', 'GL_RGBA', 'GL_HALF_FLOAT', 'NO_SWIZZLE', 'view_class_64'),
    ('R32G32B32A32_FLOAT', 'GL_RGBA32F', 'GL_RGBA', 'GL_FLOAT', 'NO_SWIZZLE', 'view_class_128'),
]

integer_rg_formats = [
    ('R8_UINT', 'GL_R8UI', 'GL_RED_INTEGER', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_8'),
    ('R8G8_UINT', 'GL_RG8UI', 'GL_RG_INTEGER', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_16'),
    ('R8_SINT', 'GL_R8I', 'GL_RED_INTEGER', 'GL_BYTE', 'NO_SWIZZLE', 'view_class_8'),
    ('R8G8_SINT', 'GL_RG8I', 'GL_RG_INTEGER', 'GL_BYTE', 'NO_SWIZZLE', 'view_class_16'),
    ('R16_UINT', 'GL_R16UI', 'GL_RED_INTEGER', 'GL_UNSIGNED_SHORT', 'NO_SWIZZLE', 'view_class_16'),
    ('R16G16_UINT', 'GL_RG16UI', 'GL_RG_INTEGER', 'GL_UNSIGNED_SHORT', 'NO_SWIZZLE', 'view_class_32'),
    ('R16_SINT', 'GL_R16I', 'GL_RED_INTEGER', 'GL_SHORT', 'NO_SWIZZLE', 'view_class_16'),
    ('R16G16_SINT', 'GL_RG16I', 'GL_RG_INTEGER', 'GL_SHORT', 'NO_SWIZZLE', 'view_class_32'),
    ('R32_UINT', 'GL_R32UI', 'GL_RED_INTEGER', 'GL_UNSIGNED_INT', 'NO_SWIZZLE', 'view_class_32'),
    ('R32G32_UINT', 'GL_RG32UI', 'GL_RG_INTEGER', 'GL_UNSIGNED_INT', 'NO_SWIZZLE', 'view_class_64'),
    ('R32_SINT', 'GL_R32I', 'GL_RED_INTEGER', 'GL_INT', 'NO_SWIZZLE', 'view_class_32'),
    ('R32G32_SINT', 'GL_RG32I', 'GL_RG_INTEGER', 'GL_INT', 'NO_SWIZZLE', 'view_class_64'),
]

float_rg_formats = [
    ('R16_FLOAT', 'GL_R16F', 'GL_RED', 'GL_HALF_FLOAT', 'NO_SWIZZLE', 'view_class_16'),
    ('R16G16_FLOAT', 'GL_RG16F', 'GL_RG', 'GL_HALF_FLOAT', 'NO_SWIZZLE', 'view_class_32'),
    ('R32_FLOAT', 'GL_R32F', 'GL_RED', 'GL_FLOAT', 'NO_SWIZZLE', 'view_class_32'),
    ('R32G32_FLOAT', 'GL_RG32F', 'GL_RG', 'GL_FLOAT', 'NO_SWIZZLE', 'view_class_64'),
]

float_3comp_formats = [
    ('R16G16B16X16_FLOAT', 'GL_RGBA16F', 'GL_RGBA', 'GL_HALF_FLOAT', 'RGB1_SWIZZLE', 'view_class_64'),
    ('R32G32B32_FLOAT', 'GL_RGB32F', 'GL_RGB', 'GL_FLOAT', 'NO_SWIZZLE', 'view_class_96'),
]

la_formats_fallback = [
    ('A8_UNORM', 'GL_R8', 'GL_RED', 'GL_UNSIGNED_BYTE', 'OOOR_SWIZZLE', 'view_class_unsupported'),
    ('L8_UNORM', 'GL_R8', 'GL_RED', 'GL_UNSIGNED_BYTE', 'RRR1_SWIZZLE', 'view_class_unsupported'),
    ('A16_UNORM', 'GL_R16', 'GL_RED', 'GL_UNSIGNED_SHORT', 'OOOR_SWIZZLE', 'view_class_unsupported'),
    ('L16_UNORM', 'GL_R16', 'GL_RED', 'GL_UNSIGNED_SHORT', 'RRR1_SWIZZLE', 'view_class_unsupported'),
    ('A8_UINT', 'GL_R8UI', 'GL_RED_INTEGER', 'GL_UNSIGNED_BYTE', 'OOOR_SWIZZLE', 'view_class_unsupported'),
    ('L8_UINT', 'GL_R8UI', 'GL_RED_INTEGER', 'GL_UNSIGNED_BYTE', 'RRR1_SWIZZLE', 'view_class_unsupported'),
    ('A8_SINT', 'GL_R8I', 'GL_RED_INTEGER', 'GL_BYTE', 'OOOR_SWIZZLE', 'view_class_unsupported'),
    ('L8_SINT', 'GL_R8I', 'GL_RED_INTEGER', 'GL_BYTE', 'RRR1_SWIZZLE', 'view_class_unsupported'),
    ('A16_UINT', 'GL_R16UI', 'GL_RED_INTEGER', 'GL_UNSIGNED_SHORT', 'OOOR_SWIZZLE', 'view_class_unsupported'),
    ('L16_UINT', 'GL_R16UI', 'GL_RED_INTEGER', 'GL_UNSIGNED_SHORT', 'RRR1_SWIZZLE', 'view_class_unsupported'),
    ('A16_SINT', 'GL_R16I', 'GL_RED_INTEGER', 'GL_SHORT', 'OOOR_SWIZZLE', 'view_class_unsupported'),
    ('L16_SINT', 'GL_R16I', 'GL_RED_INTEGER', 'GL_SHORT', 'RRR1_SWIZZLE', 'view_class_unsupported'),
    ('A32_UINT', 'GL_R32UI', 'GL_RED_INTEGER', 'GL_UNSIGNED_INT', 'OOOR_SWIZZLE', 'view_class_unsupported'),
    ('L32_UINT', 'GL_R32UI', 'GL_RED_INTEGER', 'GL_UNSIGNED_INT', 'RRR1_SWIZZLE', 'view_class_unsupported'),
    ('A32_SINT', 'GL_R32I', 'GL_RED_INTEGER', 'GL_INT', 'OOOR_SWIZZLE', 'view_class_unsupported'),
    ('L32_SINT', 'GL_R32I', 'GL_RED_INTEGER', 'GL_INT', 'RRR1_SWIZZLE', 'view_class_unsupported'),
    ('L16_FLOAT', 'GL_R16F', 'GL_RED', 'GL_HALF_FLOAT', 'RRR1_SWIZZLE', 'view_class_unsupported'),
    ('L32_FLOAT', 'GL_R32F', 'GL_RED', 'GL_FLOAT', 'RRR1_SWIZZLE', 'view_class_unsupported'),
]

la_formats_compat = [
    ('A8_UNORM', 'GL_ALPHA8', 'GL_ALPHA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('A16_UNORM', 'GL_ALPHA16', 'GL_ALPHA', 'GL_UNSIGNED_SHORT', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('A8_UINT', 'GL_ALPHA8UI_EXT', 'GL_ALPHA_INTEGER', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('L8_UINT', 'GL_LUMINANCE8UI_EXT', 'GL_LUMINANCE_INTEGER_EXT', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('L8A8_UINT', 'GL_LUMINANCE_ALPHA8UI_EXT', 'GL_LUMINANCE_ALPHA_INTEGER_EXT', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('A8_SINT', 'GL_ALPHA8I_EXT', 'GL_ALPHA_INTEGER', 'GL_BYTE', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('L8_SINT', 'GL_LUMINANCE8I_EXT', 'GL_LUMINANCE_INTEGER_EXT', 'GL_BYTE', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('L8A8_SINT', 'GL_LUMINANCE_ALPHA8I_EXT', 'GL_LUMINANCE_ALPHA_INTEGER_EXT', 'GL_BYTE', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('A16_UINT', 'GL_ALPHA16UI_EXT', 'GL_ALPHA_INTEGER', 'GL_UNSIGNED_SHORT', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('L16_UINT', 'GL_LUMINANCE16UI_EXT', 'GL_LUMINANCE_INTEGER_EXT', 'GL_UNSIGNED_SHORT', 'RRR1_SWIZZLE', 'view_class_unsupported'),
    ('L16A16_UINT', 'GL_LUMINANCE_ALPHA16UI_EXT', 'GL_LUMINANCE_ALPHA_INTEGER_EXT', 'GL_UNSIGNED_SHORT', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('A16_SINT', 'GL_ALPHA16I_EXT', 'GL_ALPHA_INTEGER', 'GL_SHORT', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('L16_SINT', 'GL_LUMINANCE16I_EXT', 'GL_LUMINANCE_INTEGER_EXT', 'GL_SHORT', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('L16A16_SINT', 'GL_LUMINANCE_ALPHA16I_EXT', 'GL_LUMINANCE_ALPHA_INTEGER_EXT', 'GL_SHORT', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('A32_UINT', 'GL_ALPHA32UI_EXT', 'GL_ALPHA_INTEGER', 'GL_UNSIGNED_INT', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('L32_UINT', 'GL_R32UI', 'GL_LUMINANCE_INTEGER_EXT', 'GL_UNSIGNED_INT', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('L32A32_UINT', 'GL_LUMINANCE_ALPHA32UI_EXT', 'GL_LUMINANCE_ALPHA_INTEGER_EXT', 'GL_UNSIGNED_INT', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('A32_SINT', 'GL_ALPHA32I_EXT', 'GL_ALPHA_INTEGER', 'GL_INT', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('L32_SINT', 'GL_LUMINANCE32I_EXT', 'GL_LUMINANCE_INTEGER_EXT', 'GL_INT', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('L32A32_SINT', 'GL_LUMINANCE_ALPHA32I_EXT', 'GL_LUMINANCE_ALPHA_INTEGER_EXT', 'GL_INT', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('A16_FLOAT', 'GL_ALPHA16F_ARB', 'GL_ALPHA', 'GL_HALF_FLOAT', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('L16_FLOAT', 'GL_LUMINANCE16F_ARB', 'GL_LUMINANCE', 'GL_HALF_FLOAT', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('L16A16_FLOAT', 'GL_LUMINANCE_ALPHA16F_ARB', 'GL_LUMINANCE_ALPHA', 'GL_HALF_FLOAT_ARB', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('A32_FLOAT', 'GL_ALPHA32F_ARB', 'GL_ALPHA', 'GL_FLOAT', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('L32_FLOAT', 'GL_LUMINANCE32F_ARB', 'GL_LUMINANCE', 'GL_FLOAT', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('L32A32_FLOAT', 'GL_LUMINANCE_ALPHA32F_ARB', 'GL_LUMINANCE_ALPHA', 'GL_FLOAT', 'NO_SWIZZLE', 'view_class_unsupported'),
]

snorm_formats = [
    ('R8_SNORM', 'GL_R8_SNORM', 'GL_RED', 'GL_BYTE', 'NO_SWIZZLE', 'view_class_8'),
    ('R8G8_SNORM', 'GL_RG8_SNORM', 'GL_RG', 'GL_BYTE', 'NO_SWIZZLE', 'view_class_16'),
    ('R8G8B8A8_SNORM', 'GL_RGBA8_SNORM', 'GL_RGBA', 'GL_BYTE', 'NO_SWIZZLE', 'view_class_32'),
    ('R8G8B8X8_SNORM', 'GL_RGBA8_SNORM', 'GL_RGBA', 'GL_BYTE', 'RGB1_SWIZZLE', 'view_class_32'),
    ('R16_SNORM', 'GL_R16_SNORM', 'GL_RED', 'GL_SHORT', 'NO_SWIZZLE', 'view_class_16'),
    ('R16G16_SNORM', 'GL_RG16_SNORM', 'GL_RG', 'GL_SHORT', 'NO_SWIZZLE', 'view_class_32'),
    ('R16G16B16A16_SNORM', 'GL_RGBA16_SNORM', 'GL_RGBA', 'GL_SHORT', 'NO_SWIZZLE', 'view_class_64'),
    ('R16G16B16X16_SNORM', 'GL_RGBA16_SNORM', 'GL_RGBA', 'GL_SHORT', 'RGB1_SWIZZLE', 'view_class_64'),
]

snorm_la_formats = [
    ('A8_SNORM', 'GL_ALPHA8_SNORM', 'GL_ALPHA', 'GL_BYTE', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('L8_SNORM', 'GL_R8_SNORM', 'GL_RED', 'GL_BYTE', 'RRR1_SWIZZLE', 'view_class_unsupported'),
    ('L8A8_SNORM', 'GL_LUMINANCE8_ALPHA8_SNORM', 'GL_LUMINANCE_ALPHA', 'GL_BYTE', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('A16_SNORM', 'GL_ALPHA16_SNORM', 'GL_ALPHA', 'GL_SHORT', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('L16_SNORM', 'GL_R16_SNORM', 'GL_RED', 'GL_SHORT', 'RRR1_SWIZZLE', 'view_class_unsupported'),
    ('L16A16_SNORM', 'GL_LUMINANCE16_ALPHA16_SNORM', 'GL_LUMINANCE_ALPHA', 'GL_SHORT', 'NO_SWIZZLE', 'view_class_unsupported'),
]

dxtn_formats = [
    ('DXT1_RGB', 'GL_COMPRESSED_RGB_S3TC_DXT1_EXT', 'GL_RGB', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_dxt1_rgb'),
    ('DXT1_RGBA', 'GL_COMPRESSED_RGBA_S3TC_DXT1_EXT', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_dxt1_rgba'),
    ('DXT3_RGBA', 'GL_COMPRESSED_RGBA_S3TC_DXT3_EXT', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_dxt3_rgba'),
    ('DXT5_RGBA', 'GL_COMPRESSED_RGBA_S3TC_DXT5_EXT', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_dxt5_rgba'),
]

dxtn_srgb_formats = [
    ('DXT1_SRGB', 'GL_COMPRESSED_SRGB_S3TC_DXT1_EXT', 'GL_RGB', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_dxt1_rgb'),
    ('DXT1_SRGBA', 'GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_dxt1_rgba'),
    ('DXT3_SRGBA', 'GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_dxt3_rgba'),
    ('DXT5_SRGBA', 'GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_dxt5_rgba'),
]

etc2_formats = [
    ('ETC2_RGB8', 'GL_COMPRESSED_RGB8_ETC2', 'GL_RGB', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_etc2_rgb'),
    ('ETC2_SRGB8', 'GL_COMPRESSED_SRGB8_ETC2', 'GL_RGB', 'GL_BYTE', 'NO_SWIZZLE', 'view_class_etc2_rgb'),
    ('ETC2_RGB8A1', 'GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_etc2_rgba'),
    ('ETC2_SRGB8A1', 'GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2', 'GL_RGBA', 'GL_BYTE', 'NO_SWIZZLE', 'view_class_etc2_rgba'),
    ('ETC2_RGBA8', 'GL_COMPRESSED_RGBA8_ETC2_EAC', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_etc2_eac_rgba'),
    ('ETC2_SRGBA8', 'GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC', 'GL_RGBA', 'GL_BYTE', 'NO_SWIZZLE', 'view_class_etc2_eac_rgba'),
    ('ETC2_R11_UNORM', 'GL_COMPRESSED_R11_EAC', 'GL_RED', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('ETC2_R11_SNORM', 'GL_COMPRESSED_SIGNED_R11_EAC', 'GL_RED', 'GL_BYTE', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('ETC2_RG11_UNORM', 'GL_COMPRESSED_RG11_EAC', 'GL_RG', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('ETC2_RG11_SNORM', 'GL_COMPRESSED_SIGNED_RG11_EAC', 'GL_RG', 'GL_BYTE', 'NO_SWIZZLE', 'view_class_unsupported'),
]

# The C's ASTC_FORMAT macro names the linear format on both of its rows, so the sRGB triple
# overwrote the linear one and the sRGB format was never registered. The second row here is the
# one the macro meant.
astc_formats = [
    ('ASTC_4x4', 'GL_COMPRESSED_RGBA_ASTC_4x4', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_astc_4x4_rgba'),
    ('ASTC_4x4_SRGB', 'GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4', 'GL_RGBA', 'GL_BYTE', 'NO_SWIZZLE', 'view_class_astc_4x4_rgba'),
    ('ASTC_5x4', 'GL_COMPRESSED_RGBA_ASTC_5x4', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_astc_5x4_rgba'),
    ('ASTC_5x4_SRGB', 'GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x4', 'GL_RGBA', 'GL_BYTE', 'NO_SWIZZLE', 'view_class_astc_5x4_rgba'),
    ('ASTC_5x5', 'GL_COMPRESSED_RGBA_ASTC_5x5', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_astc_5x5_rgba'),
    ('ASTC_5x5_SRGB', 'GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5', 'GL_RGBA', 'GL_BYTE', 'NO_SWIZZLE', 'view_class_astc_5x5_rgba'),
    ('ASTC_6x5', 'GL_COMPRESSED_RGBA_ASTC_6x5', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_astc_6x5_rgba'),
    ('ASTC_6x5_SRGB', 'GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5', 'GL_RGBA', 'GL_BYTE', 'NO_SWIZZLE', 'view_class_astc_6x5_rgba'),
    ('ASTC_6x6', 'GL_COMPRESSED_RGBA_ASTC_6x6', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_astc_6x6_rgba'),
    ('ASTC_6x6_SRGB', 'GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6', 'GL_RGBA', 'GL_BYTE', 'NO_SWIZZLE', 'view_class_astc_6x6_rgba'),
    ('ASTC_8x5', 'GL_COMPRESSED_RGBA_ASTC_8x5', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_astc_8x5_rgba'),
    ('ASTC_8x5_SRGB', 'GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x5', 'GL_RGBA', 'GL_BYTE', 'NO_SWIZZLE', 'view_class_astc_8x5_rgba'),
    ('ASTC_8x6', 'GL_COMPRESSED_RGBA_ASTC_8x6', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_astc_8x6_rgba'),
    ('ASTC_8x6_SRGB', 'GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x6', 'GL_RGBA', 'GL_BYTE', 'NO_SWIZZLE', 'view_class_astc_8x6_rgba'),
    ('ASTC_8x8', 'GL_COMPRESSED_RGBA_ASTC_8x8', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_astc_8x8_rgba'),
    ('ASTC_8x8_SRGB', 'GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x8', 'GL_RGBA', 'GL_BYTE', 'NO_SWIZZLE', 'view_class_astc_8x8_rgba'),
    ('ASTC_10x5', 'GL_COMPRESSED_RGBA_ASTC_10x5', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_astc_10x5_rgba'),
    ('ASTC_10x5_SRGB', 'GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x5', 'GL_RGBA', 'GL_BYTE', 'NO_SWIZZLE', 'view_class_astc_10x5_rgba'),
    ('ASTC_10x6', 'GL_COMPRESSED_RGBA_ASTC_10x6', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_astc_10x6_rgba'),
    ('ASTC_10x6_SRGB', 'GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x6', 'GL_RGBA', 'GL_BYTE', 'NO_SWIZZLE', 'view_class_astc_10x6_rgba'),
    ('ASTC_10x8', 'GL_COMPRESSED_RGBA_ASTC_10x8', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_astc_10x8_rgba'),
    ('ASTC_10x8_SRGB', 'GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x8', 'GL_RGBA', 'GL_BYTE', 'NO_SWIZZLE', 'view_class_astc_10x8_rgba'),
    ('ASTC_10x10', 'GL_COMPRESSED_RGBA_ASTC_10x10', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_astc_10x10_rgba'),
    ('ASTC_10x10_SRGB', 'GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x10', 'GL_RGBA', 'GL_BYTE', 'NO_SWIZZLE', 'view_class_astc_10x10_rgba'),
    ('ASTC_12x10', 'GL_COMPRESSED_RGBA_ASTC_12x10', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_astc_12x10_rgba'),
    ('ASTC_12x10_SRGB', 'GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x10', 'GL_RGBA', 'GL_BYTE', 'NO_SWIZZLE', 'view_class_astc_12x10_rgba'),
    ('ASTC_12x12', 'GL_COMPRESSED_RGBA_ASTC_12x12', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_astc_12x12_rgba'),
    ('ASTC_12x12_SRGB', 'GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x12', 'GL_RGBA', 'GL_BYTE', 'NO_SWIZZLE', 'view_class_astc_12x12_rgba'),
]

rgtc_formats = [
    ('RGTC1_UNORM', 'GL_COMPRESSED_RED_RGTC1', 'GL_RED', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_rgtc1_red'),
    ('RGTC1_SNORM', 'GL_COMPRESSED_SIGNED_RED_RGTC1', 'GL_RED', 'GL_BYTE', 'NO_SWIZZLE', 'view_class_rgtc1_red'),
    ('RGTC2_UNORM', 'GL_COMPRESSED_RG_RGTC2', 'GL_RG', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_rgtc2_rg'),
    ('RGTC2_SNORM', 'GL_COMPRESSED_SIGNED_RG_RGTC2', 'GL_RG', 'GL_BYTE', 'NO_SWIZZLE', 'view_class_rgtc2_rg'),
]

srgb_formats = [
    ('R8G8B8X8_SRGB', 'GL_SRGB8_ALPHA8', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'RGB1_SWIZZLE', 'view_class_32'),
    ('R8G8B8A8_SRGB', 'GL_SRGB8_ALPHA8', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_32'),
    ('L8_SRGB', 'GL_SR8_EXT', 'GL_RED', 'GL_UNSIGNED_BYTE', 'RRR1_SWIZZLE', 'view_class_unsupported'),
    ('R8_SRGB', 'GL_SR8_EXT', 'GL_RED', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('R8G8_SRGB', 'GL_SRG8_EXT', 'GL_RG', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_16'),
]

bit10_formats = [
    ('B10G10R10A2_UINT', 'GL_RGB10_A2UI', 'GL_BGRA_INTEGER', 'GL_UNSIGNED_INT_2_10_10_10_REV', 'NO_SWIZZLE', 'view_class_32'),
    ('R10G10B10X2_UNORM', 'GL_RGB10_A2', 'GL_RGBA', 'GL_UNSIGNED_INT_2_10_10_10_REV', 'RGB1_SWIZZLE', 'view_class_32'),
    ('R10G10B10A2_UNORM', 'GL_RGB10_A2', 'GL_RGBA', 'GL_UNSIGNED_INT_2_10_10_10_REV', 'NO_SWIZZLE', 'view_class_32'),
    ('R10G10B10A2_UINT', 'GL_RGB10_A2UI', 'GL_RGBA_INTEGER', 'GL_UNSIGNED_INT_2_10_10_10_REV', 'NO_SWIZZLE', 'view_class_32'),
]

gl_bit10_formats = [
    ('B10G10R10X2_UNORM', 'GL_RGB10_A2', 'GL_BGRA', 'GL_UNSIGNED_INT_2_10_10_10_REV', 'RGB1_SWIZZLE', 'view_class_32'),
    ('B10G10R10A2_UNORM', 'GL_RGB10_A2', 'GL_BGRA', 'GL_UNSIGNED_INT_2_10_10_10_REV', 'NO_SWIZZLE', 'view_class_32'),
]

gles_bit10_formats = [
    ('B10G10R10X2_UNORM', 'GL_RGB10_A2', 'GL_RGBA', 'GL_UNSIGNED_INT_2_10_10_10_REV', 'RGB1_SWIZZLE', 'view_class_32'),
    ('B10G10R10A2_UNORM', 'GL_RGB10_A2', 'GL_RGBA', 'GL_UNSIGNED_INT_2_10_10_10_REV', 'NO_SWIZZLE', 'view_class_32'),
]

packed_float_formats = [
    ('R11G11B10_FLOAT', 'GL_R11F_G11F_B10F', 'GL_RGB', 'GL_UNSIGNED_INT_10F_11F_11F_REV', 'NO_SWIZZLE', 'view_class_32'),
]

exponent_float_formats = [
    ('R9G9B9E5_FLOAT', 'GL_RGB9_E5', 'GL_RGB', 'GL_UNSIGNED_INT_5_9_9_9_REV', 'NO_SWIZZLE', 'view_class_32'),
]

bptc_formats = [
    ('BPTC_RGBA_UNORM', 'GL_COMPRESSED_RGBA_BPTC_UNORM', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_bptc_unorm'),
    ('BPTC_SRGBA', 'GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_bptc_unorm'),
    ('BPTC_RGB_FLOAT', 'GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT', 'GL_RGB', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_bptc_float'),
    ('BPTC_RGB_UFLOAT', 'GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT', 'GL_RGB', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_bptc_float'),
]

gl_bgra_formats = [
    ('B8G8R8X8_UNORM', 'GL_RGBA8', 'GL_BGRA', 'GL_UNSIGNED_BYTE', 'RGB1_SWIZZLE', 'view_class_32'),
    ('B8G8R8A8_UNORM', 'GL_RGBA8', 'GL_BGRA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_32'),
    ('B8G8R8X8_SRGB', 'GL_SRGB8_ALPHA8', 'GL_BGRA', 'GL_UNSIGNED_BYTE', 'RGB1_SWIZZLE', 'view_class_32'),
    ('B8G8R8A8_SRGB', 'GL_SRGB8_ALPHA8', 'GL_BGRA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_32'),
]

gles_bgra_formats = [
    ('B8G8R8X8_UNORM', 'GL_RGBA8', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'RGB1_SWIZZLE', 'view_class_32'),
    ('B8G8R8A8_UNORM', 'GL_RGBA8', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_32'),
    ('B8G8R8X8_SRGB', 'GL_SRGB8_ALPHA8', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'RGB1_SWIZZLE', 'view_class_32'),
    ('B8G8R8A8_SRGB', 'GL_SRGB8_ALPHA8', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_32'),
]

yuv_planar_formats = [
    ('NV12', 'GL_RGBA8', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('NV21', 'GL_RGBA8', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('IYUV', 'GL_RGBA8', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_unsupported'),
    ('YV12', 'GL_RGBA8', 'GL_RGBA', 'GL_UNSIGNED_BYTE', 'NO_SWIZZLE', 'view_class_unsupported'),
]
