"use client";

import { InputAdornment } from "@mui/material";
import { Search } from "@mui/icons-material";
import { StyledTextField } from "../styled-mui-components/styled-components";


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