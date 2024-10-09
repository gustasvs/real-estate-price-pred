"use client";

import React, { useEffect, useState } from 'react';
import { Button, Divider, Modal } from 'antd';

import styles from './SignUpModal.module.css';
import { GoogleCircleFilled, GoogleOutlined } from '@ant-design/icons';

interface SignUpModalProps {
  open: boolean;
  setOpen: (open: boolean) => void;
}

const SignUpModal: React.FC<SignUpModalProps> = ({ open, setOpen }

) => {
  
  const handleCloseModal = () => {
    setOpen(false);
  };

  return (
    <>
      <Modal
        // title="Title"
        open={open}
        onCancel={handleCloseModal}
        className={styles["sign-up-modal"]}
        width={484}
        footer={null}
      >
        <div className={styles["sign-up-modal-contents"]}>
        <div className={styles["google-login-container"]}>
            <Button className={`${styles["google-login-button"]} ${styles["login-button"]}`}>
                {/* <GoogleCircleFilled className={styles["google-icon"]} /> */}
                <GoogleOutlined className={styles["google-icon"]} />
              <span
                className={`${styles["google-login-text"]} ${styles["login-text"]}`}
              >
                Reģistrēties izmantojot Google
              </span>
            </Button>
        </div>
        <div className={styles["divider-container"]}>
            <Divider plain>vai</Divider>
        </div>
        <div className={styles["email-login-container"]}>
            <Button className={`${styles["email-login-button"]} ${styles["login-button"]}`}>
              <span
                className={`${styles["email-login-text"]} ${styles["login-text"]}`}
              >Reģistrēties izmantojot e-pastu</span>
            </Button>
        </div>
        <div className={styles["terms-container"]}>
            <span className={styles["terms-text"]}>
              By signing up, you agree to our Terms & Conditions, Privacy Policy and User data & Cookie policy
            </span>
        </div>
        <div className={styles["login-container"]}>
            <span className={styles["login-text"]}>
              Vai tev jau ir konts? <a href="#">Autorizēties</a>
            </span>
        </div>

        </div>
        </Modal>
    </>
  );
};

export default SignUpModal;