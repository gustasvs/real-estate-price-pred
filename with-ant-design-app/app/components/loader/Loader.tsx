import React from 'react';
import styles from './Loader.module.css';

const Loader: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({ ...props }) => {
    return (
        <div className={styles.loader} {...props}>
        </div>
    );
};

export default Loader;