import React, { useState } from 'react';
import { Button, Divider, Modal } from 'antd';

// import styles from './SignUpModal.module.css';
// temporary fix for styles not loading
const styles = {
    "sign-up-modal": "sign-up-modal",
    "sign-up-modal-contents": "sign-up-modal-contents",
    "google-login-container": "google-login-container",
    "google-login-button": "google-login-button",
    "google-icon": "google-icon",
    "google-login-text": "google-login-text",
    "login-button": "login-button",
    "divider-container": "divider-container",
    "email-login-container": "email-login-container",
    "email-login-button": "email-login-button",
    "email-login-text": "email-login-text",
    "terms-container": "terms-container",
    "terms-text": "terms-text",
    "login-container": "login-container",
    "login-text": "login-text"
    };

import { GoogleCircleFilled, GoogleOutlined } from '@ant-design/icons';

interface SignUpModalProps {
  open: boolean;
  setOpen: (open: boolean) => void;
}

const SignUpModal: React.FC<SignUpModalProps> = ({ open, setOpen }

) => {
  
  const handleOk = () => {
    setOpen(false);
  };

  const handleCancel = () => {
    console.log('Clicked cancel button');
    setOpen(false);
  };

  return (
    <>
      <Modal
        // title="Title"
        open={open}
        onOk={handleOk}
        onClose={handleCancel}
        onCancel={handleCancel}
        destroyOnClose={true}
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
                {/* Sign up with Google */}
                Reģistrēties izmantojot Google
                </span>
            </Button>
        </div>
        <div className={styles["divider-container"]}>
            <Divider plain>or</Divider>
        </div>
        <div className={styles["email-login-container"]}>
            <Button className={`${styles["email-login-button"]} ${styles["login-button"]}`}>
              <span
                className={`${styles["email-login-text"]} ${styles["login-text"]}`}
              >
                {/* Sign up with email */}
                Reģistrēties izmantojot e-pastu
                
                </span>
            </Button>
        </div>
        <div className={styles["terms-container"]}>
            <span className={styles["terms-text"]}>
              By signing up, you agree to our Terms & Conditions, Privacy Policy and User data & Cookie policy
            </span>
        </div>
        <div className={styles["login-container"]}>
            <span className={styles["login-text"]}>
              Already have an account? <a href="#">Log in</a>
            </span>
        </div>

        </div>
        </Modal>
    </>
  );
};

export default SignUpModal;