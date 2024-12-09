"use client";

import { StyledTextField } from "../my-profile/my-profile-form/MyProfileForm";
import { InputAdornment } from "@mui/material";
import { Search } from "@mui/icons-material";


const SearchInput = (
    { placeholder, onChange, ...props }:
    { placeholder: string, onChange: any, [key: string]: any }
) => {
    return (
        <StyledTextField
            placeholder={placeholder}
            style={{
                width: "100%",
                marginTop: "1rem",
            }}
            slotProps={{
                input: {
                    startAdornment: (
                        <InputAdornment position="start">
                            <Search
                                style={{
                                    color: "var(--background-light-secondary)",
                                }}
                            />
                        </InputAdornment>
                    ),
                },
            }}
            onChange={onChange}
            {...props}
        />
    )
}

export default SearchInput;