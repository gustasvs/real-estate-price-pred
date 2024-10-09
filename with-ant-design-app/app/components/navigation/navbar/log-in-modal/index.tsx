"use client";

import React from 'react';
import { Button, Divider, Modal } from 'antd';
import styles from './LogInModal.module.css';
import { GoogleCircleFilled, GoogleOutlined } from '@ant-design/icons';

interface LoginModalProps {
  open: boolean;
  setOpen: (open: boolean) => void;
}

const LoginModal: React.FC<LoginModalProps> = ({ open, setOpen }) => {
  
  const handleCloseModal = () => {
    setOpen(false);
  };

  return (
    <>
      <Modal
        open={open}
        onCancel={handleCloseModal}
        className={styles["login-modal"]}
        width={484}
        footer={null}
      >
        <div className={styles["login-modal-contents"]}>
          <div className={styles["google-login-container"]}>
            <Button className={`${styles["google-login-button"]} ${styles["login-button"]}`}>
              <GoogleOutlined className={styles["google-icon"]} />
              <span className={`${styles["google-login-text"]} ${styles["login-text"]}`}>
                {/* Log in with Google */}
                Autorizēties izmantojot Google
              </span>
            </Button>
          </div>
          <div className={styles["divider-container"]}>
            <Divider plain>vai</Divider>
          </div>
          <div className={styles["email-login-container"]}>
            <Button className={`${styles["email-login-button"]} ${styles["login-button"]}`}>
              <span className={`${styles["email-login-text"]} ${styles["login-text"]}`}>
                {/* Log in with email */}
                Autorizēties izmantojot e-pastu
              </span>
            </Button>
          </div>
          <div className={styles["terms-container"]}>
            <span className={styles["terms-text"]}>
              By logging in, you agree to our Terms & Conditions, Privacy Policy and User data & Cookie policy
            </span>
          </div>
          <div className={styles["signup-container"]}>
            <span className={styles["signup-text"]}>
              Don't have an account? <a href="#">Sign up</a>
            </span>
          </div>
        </div>
      </Modal>
    </>
  );
};

export default LoginModal;
