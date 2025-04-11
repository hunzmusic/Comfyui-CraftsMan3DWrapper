# ComfyUI-CraftsManWrapper

This custom node package provides nodes for ComfyUI to generate 3D coarse meshes from images using the CraftsMan3D model, specifically the version utilizing DoraVAE.

**Note:** This wrapper currently only implements the *coarse mesh generation* stage described in the CraftsMan3D paper. The second-stage refinement using multi-view normal maps is not included in the underlying library code provided here.

## Features

*   **All-in-One Node:** `CraftsMan Generator (All-in-One)` for a simple image-to-mesh workflow.
*   **Staged Nodes:** For more granular control:
    *   `Load CraftsMan Pipeline`: Loads the model checkpoint.
    *   `Preprocess Image (CraftsMan)`: Prepares the input image (background removal, resizing, padding).
    *   `Sample CraftsMan Latents`: Generates the 3D shape latents using diffusion.
    *   `Decode CraftsMan Latents`: Converts latents into mesh vertices and faces.
    *   `Save CraftsMan Mesh (OBJ)`: Saves the mesh data to a file (OBJ, GLB, or PLY).

## Installation

1.  **Clone or Download:** Place the `ComfyUI-CraftsManWrapper` folder inside your `ComfyUI/custom_nodes/` directory.
2.  **Install Dependencies:** Open a terminal/command prompt, navigate to your ComfyUI installation directory (activate your virtual environment if needed), and run:
    *(Ensure `pip` corresponds to the Python environment used by ComfyUI)*.

    **For Portable/Standalone ComfyUI:** You need to run pip using the Python executable included with the portable version. Open a command prompt/terminal, navigate to this custom node's directory, and run the install command using the relative path to the embedded Python:
    ```bash
    cd ComfyUI\custom_nodes\ComfyUI-CraftsManWrapper
    ..\..\..\python_embeded\python.exe -m pip install -r requirements.txt
    ```
    *(Adjust the relative path to `python.exe` (`..\..\..`) if your portable version structure or custom node location is different)*.
3.  **Download Model:**
    *   **Option A (Recommended - Local):** Download the `config.yaml` and `model.ckpt` for the desired CraftsMan model (e.g., craftsman-DoraVAE). You can often find download links or instructions in the official model repository. For example, the original CraftsMan3D README provided these `wget` commands for the DoraVAE version:
        ```bash
        # Create directories (adjust path as needed, e.g., inside ComfyUI/models/)
        # mkdir ComfyUI/models/craftman-DoraVAE
        # cd ComfyUI/models/craftman-DoraVAE

        # Download files into your target model folder
        wget https://pub-c7137d332b4145b6b321a6c01fcf8911.r2.dev/craftsman-DoraVAE/config.yaml
        wget https://pub-c7137d332b4145b6b321a6c01fcf8911.r2.dev/craftsman-DoraVAE/model.ckpt
        ```
        Create a folder (e.g., `craftman-DoraVAE`) ideally inside your `ComfyUI/models/` directory (or alongside the `ComfyUI` folder) and place the downloaded `config.yaml` and `model.ckpt` files inside it. Then, point the `model_path` input in the node to this folder.
    *   **Option B (Hugging Face - May Require Login/Availability):** The nodes can attempt to download from a Hugging Face Hub ID (like `craftsman3d/craftsman-doravae`) if a local path isn't found or specified, but this may fail if the repository is private, gated, or not yet fully uploaded.
4.  **Restart ComfyUI:** The nodes should appear under the "generation/3d/craftsman" category (or "generation/3d" for the all-in-one node).

## Usage

### All-in-One Node

*   Connect an `IMAGE` output to the `image` input.
*   Set the `model_path` to either the local directory containing `config.yaml` and `model.ckpt` (e.g., `D:\models\craftman-DoraVAE`) or the Hugging Face Hub ID (e.g., `craftsman3d/craftsman-doravae`).
*   Adjust generation parameters (seed, steps, guidance, etc.).
*   The output `mesh_path` will be the path to the generated `.obj` file in your ComfyUI `output` directory.

### Staged Nodes

1.  **Load CraftsMan Pipeline:** Provide the `model_path` (local directory or HF ID). Outputs a `pipeline` object.
2.  **Preprocess Image (CraftsMan):** Connect the `pipeline` and an `IMAGE`. Outputs an `image_tensor` (for preview) and `image_pil` (for sampling).
3.  **Sample CraftsMan Latents:** Connect the `pipeline` and `image_pil`. Set generation parameters. Outputs `shape_latents`.
4.  **Decode CraftsMan Latents:** Connect the `pipeline` and `shape_latents`. Set `octree_depth`. Outputs `mesh_vf` (vertices/faces data).
5.  **Save CraftsMan Mesh (OBJ):** Connect `mesh_vf`. Choose `file_type` and other options. Outputs the final `mesh_path` (in the ComfyUI `output` directory).

## Notes

*   **Input Image:** Providing an image with a transparent background (RGBA) generally works best. If not provided, background removal will be attempted using `rembg`.
*   **Resolution:** The model expects images around 518x518 internally. The preprocessing node handles resizing.
*   **Memory:** 3D generation can be memory-intensive. Ensure you have sufficient VRAM/RAM.

## Acknowledgements

Based on the [CraftsMan3D](https://github.com/wyysf-98/CraftsMan) and [Dora](https://github.com/Seed3D/Dora) projects. Please cite their original work if you use these nodes in your research.
