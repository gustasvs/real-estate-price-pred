"use client";

import React, { useEffect, useState } from "react";
import { Alert, Button, Divider, Form, Input, Modal } from "antd";
import styles from "./LogInModal.module.css";
import { GithubOutlined, GoogleCircleFilled, GoogleOutlined, TwitterCircleFilled } from "@ant-design/icons";
import { login } from "../../../../../actions/auth";
import { signIn, useSession } from "next-auth/react";

interface LoginModalProps {
  open: boolean;
  setOpen: (open: boolean) => void;
  setSignUpModalOpen: (open: boolean) => void;
}

const LoginModal: React.FC<LoginModalProps> = ({ open, setOpen, setSignUpModalOpen }) => {
  const [step, setStep] = useState(0);
  const [error, setError] = useState<string | null>(null);

  interface CustomSession {
    error?: string;
  }

  const { data: session } = useSession() as { data: CustomSession | null };

  const handleCloseModal = () => {
    setOpen(false);
    setError(null);
  };

  const handleSignIn = async (values: { email: string; password: string }) => {
    const res = await signIn("credentials", {
      email: values.email,
      password: values.password,
      redirect: false, // Prevent automatic navigation on error
    });

    if (res?.error) {
      setError("Diemžēl neizdevās autorizēties. Lūdzu pārbaudi ievadītos datus un mēģiniet vēlreiz!");
    } else {
      setError(null);
    }
  };

  useEffect(() => {
    if (session && session.error)
      setError(session.error as string);
  }, [session]);

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
                    {/* Autorizēties izmantojot sociālos tīklus */}
                    Autorizējies platformā izmantojot e-pastu un paroli
                  </span>
                  {/* <div className={styles["soc-login-buttons"]}>
                  <GoogleOutlined className={`${styles["soc-login-icon"]} ${styles["google-icon"]}`} onClick={() => {
                    signIn("google", { callbackUrl: "/" });
                  }}/>

                  <TwitterCircleFilled className={styles["soc-login-icon"]} />

                  <GithubOutlined className={styles["soc-login-icon"]} />

                </div> */}
              </div>
              {/* <div className={styles["divider-container"]}>
                <Divider plain>vai</Divider>
              </div> */}
              <div className={styles["email-login-container"]}>
                <Form
                  name="basic"
                  initialValues={{ remember: true }}
                  onFinish={(values) => handleSignIn(values as { email: string; password: string })}
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

                  {error && <Alert message={error} type="error" showIcon style={{ marginTop: "1em" }} />}


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
                  Vēl nav konta? <a onClick={() => {setOpen(false); setSignUpModalOpen(true)} } className={styles["footer-link"]}>Izveidot jaunu kontu</a>
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
