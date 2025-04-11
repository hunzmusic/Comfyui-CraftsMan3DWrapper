import os
import sys
import torch
import numpy as np
import PIL.Image
import trimesh
import rembg
import folder_paths
import comfy.model_management
import comfy.utils
from omegaconf import OmegaConf
import tempfile
import hashlib

import traceback # Import traceback for detailed error logging

# Craftsman code is now expected to be within this package directory
script_directory = os.path.dirname(os.path.abspath(__file__))
comfyui_root = os.path.abspath(os.path.join(script_directory, '..', '..'))

# Explicitly add the wrapper directory to sys.path to aid import resolution
if script_directory not in sys.path:
    print(f"[ComfyUI-CraftsManWrapper] Adding node directory to sys.path: {script_directory}")
    sys.path.insert(0, script_directory)

# Try importing Craftsman - handle potential import errors
# This will now look for the 'craftsman' directory inside 'ComfyUI-CraftsManWrapper'
try:
    # Use standard imports now that craftsman is a sub-directory within the package
    import craftsman
    from craftsman.utils.config import load_config
    from craftsman.pipeline import CraftsManPipeline
    CRAFTSMAN_AVAILABLE = True
    print("[ComfyUI-CraftsManWrapper] Successfully imported bundled CraftsMan library.")
except Exception as e:
    # Catch generic Exception and print full traceback for detailed diagnosis
    print(f"[ComfyUI-CraftsManWrapper] ERROR: Failed to import the bundled CraftsMan library.")
    print(f"[ComfyUI-CraftsManWrapper] Detailed Error: {e}")
    print("[ComfyUI-CraftsManWrapper] Traceback:")
    print(traceback.format_exc())
    print(f"[ComfyUI-CraftsManWrapper] Ensure the 'craftsman' directory exists inside '{script_directory}' and all dependencies from requirements.txt are installed correctly in the ComfyUI environment.")
    CRAFTSMAN_AVAILABLE = False


# --- Helper Functions ---

def tensor_to_pil(tensor):
    """Convert ComfyUI tensor (BCHW or BHWC, float32, 0-1) to PIL Image (RGB)"""
    if tensor is None:
         raise ValueError("Input tensor is None.")
    if tensor.ndim != 4:
         raise ValueError(f"Input tensor must be 4-dimensional (Batch, H, W, C or B, C, H, W). Got ndim: {tensor.ndim}")

    # Check input format and permute if necessary to get BCHW
    if tensor.shape[1] > 4 and tensor.shape[3] in [1, 3, 4]: # BHWC format likely
        print(f"[ComfyUI-CraftsManWrapper] Detected BHWC tensor (shape: {tensor.shape}), permuting to BCHW.")
        tensor = tensor.permute(0, 3, 1, 2) # Convert BHWC to BCHW
    elif tensor.shape[1] in [1, 3, 4] and tensor.shape[2] > 4: # BCHW format likely
        print(f"[ComfyUI-CraftsManWrapper] Detected BCHW tensor (shape: {tensor.shape}).")
        pass # Already in correct channel order
    else:
        # Fallback check if dimensions are ambiguous, assume BCHW if channels dim is small
        if tensor.shape[1] in [1, 3, 4]:
             print(f"[ComfyUI-CraftsManWrapper] Assuming BCHW tensor (shape: {tensor.shape}).")
             pass
        elif tensor.shape[3] in [1, 3, 4]:
             print(f"[ComfyUI-CraftsManWrapper] Assuming BHWC tensor (shape: {tensor.shape}), permuting to BCHW.")
             tensor = tensor.permute(0, 3, 1, 2)
        else:
             raise ValueError(f"Could not determine tensor format (BCHW or BHWC). Got shape: {tensor.shape}")

    # Now tensor should be in BCHW format
    num_channels = tensor.shape[1]

    # Handle grayscale by repeating channels if necessary (now that it's BCHW)
    if num_channels == 1:
        print("[ComfyUI-CraftsManWrapper] Repeating grayscale channel to RGB.")
        tensor = tensor.repeat(1, 3, 1, 1)
        num_channels = 3 # Update channel count

    # Check final channel count
    if num_channels not in [3, 4]:
        raise ValueError(f"Tensor must have 3 (RGB) or 4 (RGBA) channels after processing. Got {num_channels} channels with shape: {tensor.shape}")

    # Assuming batch size is 1 for single image processing
    img_np = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() # Convert BCHW to HWC for PIL
    # Clamp values before converting type
    img_np = np.clip(img_np, 0.0, 1.0)
    img_np = (img_np * 255).astype(np.uint8)

    # Create PIL image with correct mode (RGB or RGBA)
    if num_channels == 4:
        print("[ComfyUI-CraftsManWrapper] tensor_to_pil: Creating RGBA PIL Image.")
        return PIL.Image.fromarray(img_np, 'RGBA')
    elif num_channels == 3:
        print("[ComfyUI-CraftsManWrapper] tensor_to_pil: Creating RGB PIL Image.")
        return PIL.Image.fromarray(img_np, 'RGB')
    else:
        # This case should ideally not be reached due to earlier checks
        raise ValueError(f"Unsupported number of channels ({num_channels}) after processing.")

# --- Global Variables ---
# Simple cache to avoid reloading the same model repeatedly in one session
# Key: (model_path, device_str, dtype_str) Value: pipeline_object
loaded_pipeline_cache = {}

# --- Staged Nodes ---

class LoadCraftsManPipeline:
    """Loads the CraftsMan pipeline object."""
    @classmethod
    def INPUT_TYPES(s):
        # Define default model path - adjust if needed
        default_model_path = "craftsman3d/craftsman-doravae"
        # Check if local model exists relative to the MAIN project dir (one level above ComfyUI)
        main_project_dir = os.path.abspath(os.path.join(comfyui_root, '..'))
        local_model_path = os.path.join(main_project_dir, 'craftman-DoraVAE')

        if os.path.isdir(local_model_path):
             default_model_path = local_model_path
             print(f"[ComfyUI-CraftsManWrapper] LoadCraftsManPipeline: Defaulting to local model path: {local_model_path}")
        else:
             print(f"[ComfyUI-CraftsManWrapper] LoadCraftsManPipeline: Local model path '{local_model_path}' not found. Defaulting to Hugging Face ID.")

        return {
            "required": {
                 "model_path": ("STRING", {"default": default_model_path, "multiline": False}),
            }
        }

    # Define a custom type name for the pipeline object if needed, or use generic
    # For simplicity, let's assume it can be passed directly for now.
    # ComfyUI might require specific handling for custom objects.
    RETURN_TYPES = ("CRAFTSMAN_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load_pipeline"
    CATEGORY = "generation/3d/craftsman"

    def load_pipeline(self, model_path):
        if not CRAFTSMAN_AVAILABLE:
             raise ImportError("CraftsMan library is not available or failed to import. Cannot proceed.")

        device = comfy.model_management.get_torch_device()
        dtype = torch.bfloat16 if comfy.model_management.should_use_bf16() else torch.float16
        cache_key = (model_path, str(device), str(dtype))

        if cache_key in loaded_pipeline_cache:
            print(f"[ComfyUI-CraftsManWrapper] LoadCraftsManPipeline: Using cached pipeline for {model_path}.")
            pipeline = loaded_pipeline_cache[cache_key]
            # Ensure model is on the correct device before returning
            pipeline.system.to(device)
            return (pipeline,)

        print(f"[ComfyUI-CraftsManWrapper] LoadCraftsManPipeline: Loading pipeline from: {model_path}")
        pbar = comfy.utils.ProgressBar(1)
        pbar.update(0)
        try:
            if not os.path.isdir(model_path):
                 print(f"[ComfyUI-CraftsManWrapper] LoadCraftsManPipeline: Model path '{model_path}' not found locally, attempting download...")

            pipeline = CraftsManPipeline.from_pretrained(
                model_path,
                device=device,
                torch_dtype=dtype
            )
            pipeline.system.to(device) # Ensure it's on device
            loaded_pipeline_cache[cache_key] = pipeline
            print(f"[ComfyUI-CraftsManWrapper] LoadCraftsManPipeline: Pipeline loaded successfully.")
            pbar.update(1)
            return (pipeline,) # Return as a tuple
        except Exception as e:
            print(f"[ComfyUI-CraftsManWrapper] LoadCraftsManPipeline: Error loading pipeline: {e}")
            if cache_key in loaded_pipeline_cache: del loaded_pipeline_cache[cache_key]
            pbar.update(1)
            raise RuntimeError(f"Failed to load CraftsMan pipeline from {model_path}: {e}")


class PreprocessImageCraftsMan:
    """Preprocesses an image for the CraftsMan pipeline."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "pipeline": ("CRAFTSMAN_PIPELINE",), # Need pipeline for its preprocess method
            },
            "optional": {
                 "foreground_ratio": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01}),
                 "force_remove_bg": ("BOOLEAN", {"default": False}),
                 "bg_red": ("INT", {"default": 255, "min": 0, "max": 255}),
                 "bg_green": ("INT", {"default": 255, "min": 0, "max": 255}),
                 "bg_blue": ("INT", {"default": 255, "min": 0, "max": 255}),
            }
        }

    # Outputting a PIL Image might be tricky to pass between nodes directly.
    # Let's output the tensor representation after processing, or maybe just keep it PIL
    # Output both standard IMAGE tensor for preview and PIL for sampling node
    RETURN_TYPES = ("IMAGE", "IMAGE_PIL",)
    RETURN_NAMES = ("image_tensor", "image_pil",)
    FUNCTION = "preprocess_image"
    CATEGORY = "generation/3d/craftsman"

    def preprocess_image(self, image, pipeline, foreground_ratio=0.95, force_remove_bg=False, bg_red=255, bg_green=255, bg_blue=255):
        if not CRAFTSMAN_AVAILABLE:
             raise ImportError("CraftsMan library is not available or failed to import. Cannot proceed.")
        if pipeline is None:
             raise ValueError("CraftsMan pipeline object is required.")

        try:
            pil_image = tensor_to_pil(image)
            print(f"[ComfyUI-CraftsManWrapper] PreprocessImage: Input image converted to PIL: {pil_image.size}")
        except Exception as e:
            raise RuntimeError(f"Failed to convert input tensor to PIL image: {e}")

        print("[ComfyUI-CraftsManWrapper] PreprocessImage: Preprocessing image...")
        try:
            # Use the preprocess_image method from the loaded pipeline
            background_color_list = [bg_red, bg_green, bg_blue]
            images_pil_processed = pipeline.preprocess_image(
                [pil_image], # Pass as a list
                force=force_remove_bg,
                background_color=background_color_list,
                foreground_ratio=foreground_ratio
            )
            processed_image_pil = images_pil_processed[0]
            print("[ComfyUI-CraftsManWrapper] PreprocessImage: Image preprocessing complete.")

            # Convert PIL back to ComfyUI Tensor (BHWC, float32, 0-1)
            img_out_np = np.array(processed_image_pil.convert("RGB")).astype(np.float32) / 255.0
            # Add batch dimension: HWC -> BHWC
            image_tensor_out = torch.from_numpy(img_out_np).unsqueeze(0)

            # Return both the tensor and the PIL image
            return (image_tensor_out, processed_image_pil,)
        except Exception as e:
            print(f"[ComfyUI-CraftsManWrapper] PreprocessImage: Error during preprocessing: {e}")
            raise RuntimeError(f"Image preprocessing failed: {e}")


class SampleCraftsManLatents:
    """Generates shape latents using the CraftsMan pipeline."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("CRAFTSMAN_PIPELINE",),
                "preprocessed_image": ("IMAGE_PIL",), # Expects PIL image from previous node
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                # eta is usually fixed in the config, but could be exposed
            }
        }

    RETURN_TYPES = ("LATENT",) # Outputting a standard ComfyUI latent structure might be best
    RETURN_NAMES = ("shape_latents",)
    FUNCTION = "sample_latents"
    CATEGORY = "generation/3d/craftsman"

    def sample_latents(self, pipeline, preprocessed_image, seed, steps, guidance_scale):
        if not CRAFTSMAN_AVAILABLE:
             raise ImportError("CraftsMan library is not available or failed to import. Cannot proceed.")
        if pipeline is None:
             raise ValueError("CraftsMan pipeline object is required.")
        if preprocessed_image is None or not isinstance(preprocessed_image, PIL.Image.Image):
             raise ValueError("Preprocessed PIL Image is required.")

        device = comfy.model_management.get_torch_device()
        # Ensure pipeline is on the correct device (might have been moved by ComfyUI)
        pipeline.system.to(device)

        print(f"[ComfyUI-CraftsManWrapper] SampleLatents: Starting latent sampling with seed={seed}, steps={steps}, guidance={guidance_scale}...")
        pbar = comfy.utils.ProgressBar(steps)
        try:
            # Prepare inputs for the system's sample method
            sample_inputs_dict = {'image': [preprocessed_image]} # Needs list of PIL images

            latents_list = pipeline.system.sample(
                sample_inputs_dict,
                sample_times=1, # Generate one latent set
                steps=steps,
                guidance_scale=guidance_scale,
                eta=0.0, # Use default eta from config
                seed=seed,
                # Progress bar update would ideally happen inside sample if possible
            )
            # Assuming sample returns a list of latents for each sample_time
            latent_result = latents_list[0] # Get the first (and only) latent tensor
            print("[ComfyUI-CraftsManWrapper] SampleLatents: Latent sampling complete.")
            pbar.update(steps)

            # Package the latent tensor in the format ComfyUI expects if necessary
            # Craftsman latent shape is likely [N, D] or [1, N, D].
            # ComfyUI standard latent is often [B, C, H, W].
            # We might need a specific format or just pass the raw tensor.
            # For now, return the raw tensor. Downstream nodes will need to handle it.
            # Let's wrap it in the standard dict format for clarity, even if shape differs.
            latent_dict = {"samples": latent_result.to(comfy.model_management.intermediate_device())}

            # Move pipeline back to CPU after use? ComfyUI should handle this.
            # comfy.model_management.unload_model_clones(pipeline.system)

            return (latent_dict,)

        except Exception as e:
            print(f"[ComfyUI-CraftsManWrapper] SampleLatents: Error during latent sampling: {e}")
            # comfy.model_management.unload_model_clones(pipeline.system)
            raise RuntimeError(f"Latent sampling failed: {e}")


class DecodeCraftsManLatents:
    """Decodes CraftsMan shape latents into mesh vertices and faces."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("CRAFTSMAN_PIPELINE",),
                "shape_latents": ("LATENT",), # Expects latent dict from Sample node
                "octree_depth": ("INT", {"default": 8, "min": 5, "max": 10}),
            }
        }

    # Output vertices and faces as tensors. Need a way to represent this.
    # Maybe define custom types or just use generic TENSOR?
    # Let's try generic TENSOR first.
    RETURN_TYPES = ("GEOMETRY_VF",) # Custom type for Vertices/Faces tuple
    RETURN_NAMES = ("mesh_vf",)
    FUNCTION = "decode_latents"
    CATEGORY = "generation/3d/craftsman"

    def decode_latents(self, pipeline, shape_latents, octree_depth):
        if not CRAFTSMAN_AVAILABLE:
             raise ImportError("CraftsMan library is not available or failed to import. Cannot proceed.")
        if pipeline is None:
             raise ValueError("CraftsMan pipeline object is required.")
        if shape_latents is None or "samples" not in shape_latents:
             raise ValueError("Valid shape latents dictionary is required.")

        device = comfy.model_management.get_torch_device()
        # Ensure pipeline is on the correct device
        pipeline.system.to(device)
        # Ensure latents are on the correct device
        latents_tensor = shape_latents["samples"].to(device)

        print(f"[ComfyUI-CraftsManWrapper] DecodeLatents: Extracting geometry with octree_depth={octree_depth}...")
        try:
            # Call the geometry extraction method
            # Note: extract_geometry might expect latents *before* z_scale_factor decoding
            # The original node calls pipeline.system.shape_model.extract_geometry(latent_result, ...)
            # where latent_result is directly from pipeline.system.sample(...)
            # Let's assume the input latents are the direct output from the sample node.
            mesh_v_f_list, has_surface_list = pipeline.system.shape_model.extract_geometry(
                latents_tensor,
                octree_depth=octree_depth,
                extract_mesh_func="mc" # Default from pipeline config
            )

            # Assuming batch size 1 from the latent input
            if not mesh_v_f_list:
                 raise RuntimeError("Geometry extraction returned no mesh data.")

            mesh_v_f = mesh_v_f_list[0] # Get data for the first item in the batch
            vertices_np, faces_np = mesh_v_f[0], mesh_v_f[1]

            print("[ComfyUI-CraftsManWrapper] DecodeLatents: Geometry extraction complete.")

            # Convert numpy arrays to tensors for output
            # Keep on the current device or move to intermediate? Let's keep on device for now.
            vertices_tensor = torch.from_numpy(vertices_np).to(device=device, dtype=torch.float32)
            faces_tensor = torch.from_numpy(faces_np).to(device=device, dtype=torch.int64) # Faces are usually int indices

            # Package as a tuple or dict for the custom type
            mesh_data = {"vertices": vertices_tensor, "faces": faces_tensor}

            # Move pipeline back to CPU? ComfyUI should handle this.
            # comfy.model_management.unload_model_clones(pipeline.system)

            return (mesh_data,) # Return the dict containing tensors

        except Exception as e:
            print(f"[ComfyUI-CraftsManWrapper] DecodeLatents: Error during geometry extraction: {e}")
            # comfy.model_management.unload_model_clones(pipeline.system)
            raise RuntimeError(f"Geometry extraction failed: {e}")


class SaveCraftsManMesh:
    """Saves mesh data (vertices, faces) to an OBJ file."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh_vf": ("GEOMETRY_VF",), # Expects dict {"vertices": tensor, "faces": tensor}
            },
            "optional": {
                "only_max_component": ("BOOLEAN", {"default": False}),
                "output_filename_prefix": ("STRING", {"default": "craftsman_mesh"}),
                "file_type": (["obj", "glb", "ply"], {"default": "obj"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("mesh_path",)
    FUNCTION = "save_mesh"
    CATEGORY = "generation/3d/craftsman"

    def save_mesh(self, mesh_vf, only_max_component=False, output_filename_prefix="craftsman_mesh", file_type="obj"):
        if not CRAFTSMAN_AVAILABLE:
             raise ImportError("CraftsMan library is not available or failed to import. Cannot proceed.")
        if mesh_vf is None or "vertices" not in mesh_vf or "faces" not in mesh_vf:
             raise ValueError("Valid mesh data dictionary {'vertices': tensor, 'faces': tensor} is required.")

        vertices_tensor = mesh_vf["vertices"]
        faces_tensor = mesh_vf["faces"]

        # Convert tensors back to numpy for trimesh, ensure they are on CPU
        vertices_np = vertices_tensor.cpu().numpy()
        faces_np = faces_tensor.cpu().numpy()

        print("[ComfyUI-CraftsManWrapper] SaveMesh: Creating Trimesh object...")
        try:
            mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np)

            # --- Post-processing (copied from original node) ---
            if only_max_component:
                print("[ComfyUI-CraftsManWrapper] SaveMesh: Splitting mesh components...")
                components = mesh.split(only_watertight=False)
                if components:
                    bbox_sizes = []
                    for c in components:
                        if c.vertices.shape[0] > 0:
                            bbmin = c.vertices.min(0)
                            bbmax = c.vertices.max(0)
                            bbox_sizes.append((bbmax - bbmin).max())
                        else:
                            bbox_sizes.append(0)

                    if bbox_sizes:
                         max_component_idx = np.argmax(bbox_sizes)
                         mesh = components[max_component_idx]
                         print(f"[ComfyUI-CraftsManWrapper] SaveMesh: Kept largest component ({max_component_idx+1}/{len(components)}).")
                    else:
                         print("[ComfyUI-CraftsManWrapper] SaveMesh: Warning: No valid components found after split.")
                else:
                     print("[ComfyUI-CraftsManWrapper] SaveMesh: Warning: Mesh split resulted in no components.")

            # Fix normals and add basic material
            print("[ComfyUI-CraftsManWrapper] SaveMesh: Fixing normals...")
            mesh.fix_normals()
            mesh.visual = trimesh.visual.TextureVisuals(
                material=trimesh.visual.material.PBRMaterial(
                    baseColorFactor=[255, 255, 255, 255], # RGBA
                    metallicFactor=0.05,
                    roughnessFactor=1.0
                )
            )

            # --- Saving Output ---
            output_dir = folder_paths.get_output_directory() # Save to standard output dir
            os.makedirs(output_dir, exist_ok=True)

            # Create a unique filename
            # Use hash of vertices/faces data for more deterministic naming if needed
            try:
                vf_hash = hashlib.sha256(vertices_np.tobytes() + faces_np.tobytes()).hexdigest()[:8]
            except:
                vf_hash = "nodata"
            # Use selected file type for extension
            file_type_clean = file_type.lower().strip()
            if file_type_clean not in ["obj", "glb", "ply"]:
                print(f"[ComfyUI-CraftsManWrapper] SaveMesh: Warning - Invalid file_type '{file_type}', defaulting to 'obj'.")
                file_type_clean = "obj"
            filename = f"{output_filename_prefix}_{vf_hash}.{file_type_clean}"
            filepath = os.path.join(output_dir, filename)

            print(f"[ComfyUI-CraftsManWrapper] SaveMesh: Exporting mesh to: {filepath} (Format: {file_type_clean})")
            mesh.export(filepath, include_normals=True, file_type=file_type_clean)
            print(f"[ComfyUI-CraftsManWrapper] SaveMesh: Mesh exported successfully.")

            return (filepath,)

        except Exception as e:
            print(f"[ComfyUI-CraftsManWrapper] SaveMesh: Error during mesh processing or saving: {e}")
            raise RuntimeError(f"Mesh processing or saving failed: {e}")


# --- Original All-in-One Node ---

class CraftsManDoraVAEGenerator:
    """
    ComfyUI node for generating 3D meshes using CraftsMan with DoraVAE.
    Generates a coarse mesh based on an input image.
    """
    @classmethod
    def INPUT_TYPES(s):
        # Define default model path - adjust if needed
        default_model_path = "craftsman3d/craftsman-doravae"
        # Check if local model exists relative to the MAIN project dir (one level above ComfyUI)
        # This allows users to keep large models outside the custom_nodes folder
        main_project_dir = os.path.abspath(os.path.join(comfyui_root, '..'))
        local_model_path = os.path.join(main_project_dir, 'craftman-DoraVAE')

        if os.path.isdir(local_model_path):
             default_model_path = local_model_path
             print(f"[ComfyUI-CraftsManWrapper] Defaulting to local model path: {local_model_path}")
        else:
             print(f"[ComfyUI-CraftsManWrapper] Local model path '{local_model_path}' not found. Defaulting to Hugging Face ID.")


        return {
            "required": {
                "image": ("IMAGE",),
                "model_path": ("STRING", {"default": default_model_path, "multiline": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "octree_depth": ("INT", {"default": 8, "min": 5, "max": 10}), # Higher depth = more detail/memory
            },
            "optional": {
                "foreground_ratio": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01}),
                "force_remove_bg": ("BOOLEAN", {"default": False}),
                "only_max_component": ("BOOLEAN", {"default": False}),
                "output_filename_prefix": ("STRING", {"default": "craftsman_mesh"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("mesh_path",)
    FUNCTION = "generate_mesh"
    CATEGORY = "generation/3d" # You might want to change this category

    def load_pipeline(self, model_path, device, dtype):
        """Loads the pipeline, using cache if possible."""
        global loaded_pipeline_cache
        cache_key = (model_path, str(device), str(dtype))

        if cache_key in loaded_pipeline_cache:
            print(f"[ComfyUI-CraftsManWrapper] Using cached pipeline for {model_path}.")
            # Ensure model is on the correct device
            pipeline = loaded_pipeline_cache[cache_key]
            pipeline.system.to(device) # Move system (main model part) to device
            return pipeline

        print(f"[ComfyUI-CraftsManWrapper] Loading CraftsMan pipeline from: {model_path}")
        # Unload other models managed by ComfyUI if necessary to free VRAM
        # comfy.model_management.unload_all_models() # Be cautious with this, might impact other nodes

        # Create a progress bar context
        pbar = comfy.utils.ProgressBar(1)
        pbar.update(0) # Initialize

        try:
            # Check if path exists locally before trying HF Hub
            if not os.path.isdir(model_path):
                 print(f"[ComfyUI-CraftsManWrapper] Model path '{model_path}' not found locally, attempting download from Hugging Face Hub...")

            pipeline = CraftsManPipeline.from_pretrained(
                model_path,
                device=device, # Load directly to target device if possible
                torch_dtype=dtype
            )
            # Ensure the entire system is on the correct device after loading
            pipeline.system.to(device)
            loaded_pipeline_cache[cache_key] = pipeline
            print(f"[ComfyUI-CraftsManWrapper] Pipeline loaded successfully.")
            pbar.update(1) # Mark as complete
            return pipeline
        except Exception as e:
            print(f"[ComfyUI-CraftsManWrapper] Error loading pipeline: {e}")
            # Clean cache entry if loading failed
            if cache_key in loaded_pipeline_cache:
                del loaded_pipeline_cache[cache_key]
            pbar.update(1) # Ensure progress bar finishes
            raise RuntimeError(f"Failed to load CraftsMan pipeline from {model_path}: {e}")


    def generate_mesh(self, image, model_path, seed, steps, guidance_scale, octree_depth,
                      foreground_ratio=0.95, force_remove_bg=False, only_max_component=False,
                      output_filename_prefix="craftsman_mesh"):

        if not CRAFTSMAN_AVAILABLE:
             raise ImportError("CraftsMan library is not available or failed to import. Cannot proceed.")

        # Determine device and dtype
        device = comfy.model_management.get_torch_device()
        dtype = torch.bfloat16 if comfy.model_management.should_use_bf16() else torch.float16

        # --- Model Loading ---
        pipeline = self.load_pipeline(model_path, device, dtype)

        # --- Input Processing ---
        try:
            pil_image = tensor_to_pil(image)
            print(f"[ComfyUI-CraftsManWrapper] Input image converted to PIL: {pil_image.size}")
        except ValueError as e:
            raise ValueError(f"Invalid input image tensor: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to convert input tensor to PIL image: {e}")


        # --- Generation ---
        print(f"[ComfyUI-CraftsManWrapper] Starting mesh generation with seed={seed}, steps={steps}, guidance={guidance_scale}...")
        # Create a progress bar for generation steps
        pbar = comfy.utils.ProgressBar(steps)
        try:
            # Note: CraftsManPipeline __call__ expects a single image or path, not batch
            # We processed the batch tensor to a single PIL image above

            # --- Preprocessing (integrated from pipeline.__call__) ---
            # This replicates the preprocessing logic inside the pipeline's __call__
            # to potentially show progress or allow finer control if needed.
            # Alternatively, just call pipeline() directly.
            print("[ComfyUI-CraftsManWrapper] Preprocessing image...")
            images_pil_processed = pipeline.preprocess_image(
                [pil_image], # Pass as a list
                force=force_remove_bg,
                background_color=[255, 255, 255], # Default background
                foreground_ratio=foreground_ratio
            )
            processed_image_for_sampling = images_pil_processed[0]
            print("[ComfyUI-CraftsManWrapper] Image preprocessing complete.")

            # --- Sampling Latents ---
            print("[ComfyUI-CraftsManWrapper] Sampling latents...")
            # Directly call system.sample for potentially more control or progress reporting
            sample_inputs_dict = {'image': [processed_image_for_sampling]} # Use the preprocessed image
            latents = pipeline.system.sample(
                sample_inputs_dict,
                sample_times=1, # Generate one mesh
                steps=steps,
                guidance_scale=guidance_scale,
                eta=0.0, # Default eta from pipeline config
                seed=seed,
                # Add progress bar update here if sample method supports callbacks
                # For now, we update after the whole sample call
            )
            # Assuming sample returns a list of latents for each sample_time
            latent_result = latents[0]
            print("[ComfyUI-CraftsManWrapper] Latent sampling complete.")
            pbar.update(steps) # Mark sampling as complete for progress

            # --- Extracting Geometry ---
            print("[ComfyUI-CraftsManWrapper] Extracting geometry...")
            mesh_v_f, has_surface = pipeline.system.shape_model.extract_geometry(
                latent_result,
                octree_depth=octree_depth,
                extract_mesh_func="mc" # Default from pipeline config
            )
            print("[ComfyUI-CraftsManWrapper] Geometry extraction complete.")

            # --- Post-processing (Trimesh) ---
            print("[ComfyUI-CraftsManWrapper] Post-processing mesh...")
            mesh = trimesh.Trimesh(vertices=mesh_v_f[0][0], faces=mesh_v_f[0][1])
            if only_max_component:
                print("[ComfyUI-CraftsManWrapper] Splitting mesh components...")
                components = mesh.split(only_watertight=False)
                if components:
                    bbox_sizes = []
                    for c in components:
                        if c.vertices.shape[0] > 0:
                            bbmin = c.vertices.min(0)
                            bbmax = c.vertices.max(0)
                            bbox_sizes.append((bbmax - bbmin).max())
                        else:
                            bbox_sizes.append(0)

                    if bbox_sizes:
                         max_component_idx = np.argmax(bbox_sizes)
                         mesh = components[max_component_idx]
                         print(f"[ComfyUI-CraftsManWrapper] Kept largest component ({max_component_idx+1}/{len(components)}).")
                    else:
                         print("[ComfyUI-CraftsManWrapper] Warning: No valid components found after split.")
                else:
                     print("[ComfyUI-CraftsManWrapper] Warning: Mesh split resulted in no components.")


            # Fix normals and add basic material (like in pipeline)
            mesh.fix_normals()
            mesh.visual = trimesh.visual.TextureVisuals(
                material=trimesh.visual.material.PBRMaterial(
                    baseColorFactor=[255, 255, 255, 255], # RGBA
                    metallicFactor=0.05,
                    roughnessFactor=1.0
                )
            )
            print(f"[ComfyUI-CraftsManWrapper] Mesh generation and post-processing complete.")

        except Exception as e:
             print(f"[ComfyUI-CraftsManWrapper] Error during mesh generation: {e}")
             # Attempt to move model off GPU before raising error
             comfy.model_management.unload_model_clones(pipeline.system)
             raise RuntimeError(f"Mesh generation failed: {e}")

        # --- Saving Output ---
        output_dir = folder_paths.get_output_directory() # Save to standard output dir
        os.makedirs(output_dir, exist_ok=True)

        # Create a unique filename based on inputs to avoid collisions
        try:
            img_hash = hashlib.sha256(image.cpu().numpy().tobytes()).hexdigest()[:8]
        except:
            img_hash = "noimg" # Fallback if hashing fails
        inputs_hash = hashlib.md5(f"{model_path}{seed}{steps}{guidance_scale}{octree_depth}{foreground_ratio}{force_remove_bg}{only_max_component}{img_hash}".encode()).hexdigest()[:8]
        filename = f"{output_filename_prefix}_{inputs_hash}.obj"
        filepath = os.path.join(output_dir, filename)

        try:
            print(f"[ComfyUI-CraftsManWrapper] Exporting mesh to: {filepath}")
            mesh.export(filepath, include_normals=True, file_type='obj')
            print(f"[ComfyUI-CraftsManWrapper] Mesh exported successfully.")
        except Exception as e:
            print(f"[ComfyUI-CraftsManWrapper] Error exporting mesh: {e}")
            raise RuntimeError(f"Failed to export mesh: {e}")
        finally:
            # Move model back to CPU after use to free VRAM (handled by ComfyUI's model management ideally)
            # comfy.model_management.unload_model_clones(pipeline.system) # Let ComfyUI manage this
            pass


        return (filepath,)

# --- Node Mappings ---
# Make sure the node appears in the ComfyUI menus
NODE_CLASS_MAPPINGS = {
    "CraftsManDoraVAEGenerator": CraftsManDoraVAEGenerator,
    "LoadCraftsManPipeline": LoadCraftsManPipeline,
    "PreprocessImageCraftsMan": PreprocessImageCraftsMan,
    "SampleCraftsManLatents": SampleCraftsManLatents,
    "DecodeCraftsManLatents": DecodeCraftsManLatents,
    "SaveCraftsManMesh": SaveCraftsManMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CraftsManDoraVAEGenerator": "CraftsMan Generator (All-in-One)",
    "LoadCraftsManPipeline": "Load CraftsMan Pipeline",
    "PreprocessImageCraftsMan": "Preprocess Image (CraftsMan)",
    "SampleCraftsManLatents": "Sample CraftsMan Latents",
    "DecodeCraftsManLatents": "Decode CraftsMan Latents",
    "SaveCraftsManMesh": "Save CraftsMan Mesh (OBJ)",
}

print("[ComfyUI-CraftsManWrapper] Loaded CraftsMan custom nodes.")
