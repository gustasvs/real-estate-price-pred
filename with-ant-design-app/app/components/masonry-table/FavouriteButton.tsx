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

const FavouriteButton = ({ favourite, onClick }: { onClick: (e: any) => void; favourite: boolean }
) => {
    
    return (
        <div
            onClick={(e) => {
                onClick(e);
                if (favourite) {
                    return;
                }
                confetti({
                    particleCount: 20,
                    // colors: ['#FF0000', '#00FF00', '#0000FF'],
                    spread: 90,
                    decay: 0.82,
                    startVelocity: 25,
                    ticks: 65,
                    origin: { x: e.clientX / window.innerWidth, y: e.clientY / window.innerHeight }
                });
            }}
            className={`${styles[
                "content-description-action"
            ]
                } ${styles[
                "content-description-action-favourite"
                ]
                } ${favourite
                    ? styles[
                    "content-description-action-favourite-active"
                    ]
                    : ""
                }`}
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