"use client";

import { useCallback, useState } from "react";
import styles from "./FavouriteButton.module.css";

import confetti from 'canvas-confetti';


import {
    EditOutlined,
    HeartFilled,
    HeartOutlined,
    PlusOutlined,
} from "@ant-design/icons";

const FavouriteButton = ({ 
    favourite, 
    onClick,
    fromForm=false,
    ...props 
}: { 
    onClick: (e: any) => void; 
    favourite: boolean;
    fromForm: boolean; 
    [key: string]: any 
}) => {

    return (
        <div
            onClick={(e) => {
                onClick(e);
                if (favourite) {
                    return;
                }
                confetti({
                    particleCount: 20,
                    colors: ['#ffb341', '#b36b00', '#ffd699'],
                    spread: 90,
                    decay: 0.82,
                    startVelocity: 25,
                    ticks: 65,
                    origin: { x: e.clientX / window.innerWidth, y: e.clientY / window.innerHeight }
                });
            }}
            className={`${styles["content-description-action"]} 
            ${styles["content-description-action-favourite"]} 
            ${favourite ? styles["content-description-action-favourite-active"]: ""}
            ${fromForm ? styles["content-description-action-favourite-form"]: ""}
            `}
            {...props}
        >
            {favourite ? (
                <HeartFilled />
            ) : (
                <HeartOutlined />
            )}

        </div>
    );
};

export default FavouriteButton;