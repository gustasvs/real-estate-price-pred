"use client";

import React from "react";
import { Button, Divider, Form, Input, Modal } from "antd";
import styles from "./LogInModal.module.css";
import { GithubOutlined, GoogleCircleFilled, GoogleOutlined, TwitterCircleFilled } from "@ant-design/icons";
import { login } from "../../../../../actions/auth";
import { signIn } from "next-auth/react";

interface LoginModalProps {
  open: boolean;
  setOpen: (open: boolean) => void;
}

const LoginModal: React.FC<LoginModalProps> = ({ open, setOpen }) => {
  const handleCloseModal = () => {
    setOpen(false);
  };

  const [step, setStep] = React.useState(0);

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
          {step === 0 && (
            <>
              <div className={styles["soc-login-container"]}>
                <span
                    className={styles["soc-login-text"]}
                  >
                    Autorizēties izmantojot sociālos tīklus
                  </span>
                  <div className={styles["soc-login-buttons"]}>
                  <GoogleOutlined className={`${styles["soc-login-icon"]} ${styles["google-icon"]}`} onClick={() => {
                    signIn("google", { callbackUrl: "/" });
                  }}/>

                  <TwitterCircleFilled className={styles["soc-login-icon"]} />

                  <GithubOutlined className={styles["soc-login-icon"]} />

                </div>
              </div>
              <div className={styles["divider-container"]}>
                <Divider plain>vai</Divider>
              </div>
              <div className={styles["email-login-container"]}>
                <Form
                  name="basic"
                  initialValues={{ remember: true }}
                  onFinish={(values) => {
                    console.log("Received values of form: ", values);
                    signIn("credentials", {
                      email: values.email,
                      password: values.password,
                      callbackUrl: "/",
                    });
                  }}
                  style={{ width: "100%" }}
                >
                  <Form.Item
                    name="email"
                    rules={[{ required: true, message: "Lūdzu ievadiet e-pastu!" }]}
                    style={{ marginBottom: "1em" }}
                  >
                    <Input placeholder="E-pasts" className={styles["email-input"]} />
                  </Form.Item>

                  <Form.Item
                    name="password"
                    rules={[{ required: true, message: "Lūdzu ievadiet paroli!" }]}
                    className={styles["password-form-item"]}
                    style={{ margin: 0 }}
                  >
                    <Input.Password placeholder="Parole" className={styles["password-input"]} />
                  </Form.Item>

                  <div className={styles["button-container"]}>
                    <Button type="primary" htmlType="submit" className={styles["submit-button"]}>
                      Autorizēties
                    </Button>
                  </div>
                </Form>
              </div>
              <div className={styles["terms-container"]}>
                <span className={styles["terms-text"]}>
                  {/* By logging in, you agree to our Terms & Conditions, Privacy Policy and User data & Cookie policy */}
                  Autorizējoties jūs piekrītat portāla Lietošanas noteikumiem,
                  Konfidencialitātes politikai un Lietotāja datu un sīkdatņu
                  politikai
                </span>
              </div>
              <div className={styles["signup-container"]}>
                <span className={styles["signup-text"]}>
                  {/* Don't have an account? <a href="#">Sign up</a> */}
                  Vēl nav konta? <a href="#">Reģistrēties</a>
                </span>
              </div>
            </>
          )}
          
        </div>
      </Modal>
    </>
  );
};

export default LoginModal;
