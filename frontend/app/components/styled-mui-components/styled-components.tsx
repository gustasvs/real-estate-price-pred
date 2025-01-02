import { IconButton, Slider, styled, TextField } from "@mui/material";
import Switch from '@mui/material/Switch';


export const StyledTextField = styled(TextField)({

    '& .MuiTextField-root': {
        // margin: '1em 0',
        width: '100%',
    },

    '& .MuiInputBase-root': {
        backgroundColor: "var(--background-dark-main-hover)",
        outlineColor: "var(--background-light-main)",
    },

    '& .MuiOutlinedInput-root fieldset': {
        borderColor: "var(--background-light-main) !important",
    },

    '& label': {
        color: "var(--background-light-main)",
    },

    '& input': {
        color: "white",
    },

    '& label.Mui-focused': {
        color: "var(--background-light-main)",
    },
    '& .MuiInput-underline:after': {
        borderBottomColor: '#B2BAC2',
    },
    '& .MuiOutlinedInput-root': {
        borderRadius: '1em',
        '& fieldset': {
            borderColor: '#E0E3E7',
        },
        '&:hover fieldset': {
            borderColor: '#B2BAC2',
        },
        '&.Mui-focused fieldset': {
            borderColor: '#6F7E8C',
        },
        // Error state
        '&.Mui-error fieldset': {
            borderColor: 'red', // Red border for error state
        },
        '&.Mui-error:hover fieldset': {
            borderColor: 'darkred', // Darker red border on hover in error state
        },
        '&.Mui-error.Mui-focused fieldset': {
            borderColor: 'darkred', // Darker red border when focused in error state
        },
    },
    // palceholder color
    '& .MuiOutlinedInput-input': {
        color: "var(--text-brighter)",
    },

    '&.Mui-error fieldset': {
        borderColor: 'red', // This line adds the red border when there is an error
    },

    // helper text color
    '& .MuiFormHelperText-root': {
        textAlign: 'right',
        color: "var(--background-light-main)",
    },

    '& .MuiFilledInput-root::after': {
        borderBottomColor: 'var(--background-light-main)',
    },
    '& .MuiFilledInput-root': {
        backgroundColor: 'var(--background-dark-main-hover)',
        // outline: '1px solid var(--background-light-main)',
        borderColor: 'var(--background-light-main)',
    },

    '& .MuiOutlinedInput-notchedOutline label': {
        width: 'unset',
    },

});

export const StyledNumberInput = styled(TextField)({

    '& .MuiInputBase-root': {
        backgroundColor: "var(--background-dark-main-hover)",
    },

    '& .MuiOutlinedInput-root fieldset': {
        borderColor: "var(--background-light-main) !important",
    },

    '& label': {
        color: "var(--background-light-main)",
    },

    '& input': {
        color: "var(--background-light-secondary)",
        MozAppearance: 'textfield', // Removes arrows in Firefox
    },

    '& input[type=number]': {
        WebkitAppearance: 'none', // Removes arrows in Chrome
        margin: 0, // Ensures consistency in alignment
    },

    '& label.Mui-focused': {
        color: "var(--background-light-main)",
    },

    '& .MuiInput-underline:after': {
        borderBottomColor: '#B2BAC2',
    },

    '& .MuiOutlinedInput-root': {
        borderRadius: '1em',
        '& fieldset': {
            borderColor: '#E0E3E7',
        },
        '&:hover fieldset': {
            borderColor: '#B2BAC2',
        },
        '&.Mui-focused fieldset': {
            borderColor: '#6F7E8C',
        },
    },
});


export const StyledSlider = styled(Slider)({
    color: "var(--background-light-main)", // Primary color
    height: 8,
    '& .MuiSlider-thumb': {
        height: 24,
        width: 24,
        backgroundColor: "var(--background-dark-main-hover)",
        border: "2px solid var(--background-light-main)",
        '&:hover': {
            boxShadow: "0 0 0 8px rgba(255, 255, 255, 0.16)", // Subtle hover effect
        },
        '&.Mui-focusVisible': {
            boxShadow: "0 0 0 8px rgba(255, 255, 255, 0.24)", // Focus effect
        },
        '&.Mui-active': {
            boxShadow: "0 0 0 12px rgba(255, 255, 255, 0.32)", // Active effect
        },
    },
    '& .MuiSlider-rail': {
        height: 8,
        backgroundColor: "var(--background-dark-secondary)", // Rail color
    },
    '& .MuiSlider-track': {
        height: 8,
        backgroundColor: "var(--background-light-secondary)", // Track color
        border: "none",
    },
    '& .MuiSlider-mark': {
        backgroundColor: "var(--text-brighter)", // Mark color
        height: 8,
        width: 8,
        borderRadius: '50%',
    },
    '& .MuiSlider-markActive': {
        backgroundColor: "var(--background-light-main)", // Active mark color
    },
    '& .MuiSlider-valueLabelOpen': {
        borderRadius: '10px',
    },
});


export const StyledSwitch = styled(Switch)(({ theme }) => ({
    '& .MuiSwitch-switchBase': {
        color: "var(--background-light-secondary)",
        '&:hover': {
            backgroundColor: 'rgba(0, 0, 0, 0.1)',
        },
        '&.Mui-checked': {
            color: "var(--background-light-main)",
            '&:hover': {
                backgroundColor: 'rgba(0, 0, 0, 0.2)',
            },
        },
    },
    '& .MuiSwitch-track': {
        backgroundColor: '#E0E3E7',
        opacity: 1,
        borderRadius: 16,
    },
    '& .Mui-checked + .MuiSwitch-track': {
        backgroundColor: '#6F7E8C',
        opacity: 1,
    },
}));


export const StyledIconButton = styled(IconButton)({
    '& .MuiSvgIcon-root': {
      fill: "inherit",
    },
    
    '& .MuiIconButton-root:hover': {
      backgroundColor: "var(--background-light-main)",
      color: "var(--background-light-main)",
    }
  });