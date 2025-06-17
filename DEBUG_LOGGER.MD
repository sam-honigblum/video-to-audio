# Debug Log: CAVP Model Loading Issue

## Issue Description
- Error when loading CAVP model checkpoint due to state dictionary key mismatches
- Missing keys: "audio_encoder.bn.weight", "audio_encoder.bn.bias", etc.
- Unexpected keys: "audio_encoder.bn0.weight", "audio_encoder.bn0.bias", etc.

## Root Cause
The checkpoint was saved with an older version of the model architecture where the audio encoder used different layer naming conventions (e.g., "bn0" instead of "bn").

## Solution Implemented
1. Modified CAVP_VideoOnly class to handle state dictionary mismatch:
   - Added key mapping to convert old layer names to new ones
   - Specifically mapped:
     - "bn0" → "bn"
     - "fc_audioset" → "project_head"
   - Used strict=False when loading state dict to handle any remaining mismatches

## Files Modified
1. src/models/cavp_encoder.py
   - Updated CAVP_VideoOnly.__init__ method
   - Added state dictionary key mapping logic
   - Maintained existing functionality while adding compatibility layer

## Next Steps
1. Test the modified code with the checkpoint
2. Verify that the model loads correctly
3. Monitor for any remaining state dict mismatches
