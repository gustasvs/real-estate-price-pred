"use client";

import styles from "./returnButton.module.css";

export const ReturnButton = ({ text }: { text: string }) => {
    
    return (
        <button
            onClick={() => {
                window.location.href = "/";
            }}
            className={styles.returnButton}
        >
            {text}
        </button>
    );
}


export default ReturnButton;