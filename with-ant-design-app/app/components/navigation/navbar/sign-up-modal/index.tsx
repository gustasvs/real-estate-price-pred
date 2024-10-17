import React, { useState } from "react";
import { Alert, Button, Divider, Form, Input, Modal } from "antd";
import {
  GithubOutlined,
  GoogleOutlined,
  TwitterCircleFilled,
} from "@ant-design/icons";
import styles from "./SignUpModal.module.css";
import { signIn } from "next-auth/react";

interface SignUpModalProps {
  open: boolean;
  setOpen: (open: boolean) => void;
}

const SignUpModal: React.FC<SignUpModalProps> = ({ open, setOpen }) => {
  const [error, setError] = useState<string | null>(null);

  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  const handleCloseModal = () => {
    setOpen(false);
  };

  const passwordStrength = (password: string) => {
    if (password.length < 8) {
      return 0;
    }
    return 1;
  };

  const handlePasswordChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (passwordStrength(event.target.value)) {
      setShowConfirmPassword(true);
    } else {
      setShowConfirmPassword(false);
    }
  };

  const handleSignUp = async (values: {
    email: string;
    password: string;
    confirmPassword: string;
  }) => {
    if (values.password !== values.confirmPassword) {
      setError("Passwords do not match. Please try again.");
      return;
    }
    
    await signIn("credentials", {
      email: values.email,
      password: values.password,
      confirmPassword: values.confirmPassword,
      redirect: false, // Prevent automatic navigation on error
    });
    
    setError(null);
  };

  return (
    <>
      <Modal
        open={open}
        onCancel={handleCloseModal}
        className={styles["sign-up-modal"]}
        width={484}
        footer={null}
      >
        <div className={styles["sign-up-modal-contents"]}>
          <div className={styles["soc-login-container"]}>
            <span className={styles["soc-login-text"]}>
              Izveido kontu izmantojot sociālos tīklus
            </span>
            <div className={styles["soc-login-buttons"]}>
              <GoogleOutlined
                className={`${styles["soc-login-icon"]} ${styles["google-icon"]}`}
                onClick={() => {
                  // signIn("google", { callbackUrl: "/" });
                }}
              />

              <TwitterCircleFilled className={styles["soc-login-icon"]} />

              <GithubOutlined className={styles["soc-login-icon"]} />
            </div>
          </div>
          <div className={styles["divider-container"]}>
            <Divider plain>vai</Divider>
          </div>
          <div className={styles["email-sign-up-container"]}>
            <Form
              name="basic"
              initialValues={{ remember: true }}
              onFinish={handleSignUp}
              style={{ width: "100%" }}
            >
              <Form.Item
                name="email"
                rules={[{ required: true, message: "Lūdzu ievadiet e-pastu!" }]}
                style={{ marginBottom: "1em" }}
              >
                <Input
                  placeholder="E-pasts"
                  className={styles["email-input"]}
                />
              </Form.Item>

              <Form.Item
                name="password"
                rules={[{ required: true, message: "Lūdzu ievadiet paroli!" }]}
                className={styles["password-form-item"]}
                style={{ marginBottom: "1em" }}
              >
                <Input.Password
                  placeholder="Parole"
                  className={styles["password-input"]}
                  onChange={handlePasswordChange}
                />
              </Form.Item>
              <div
                className={`${styles["confirm-password-container"]} ${
                  showConfirmPassword ? styles.visible : ""
                }`}
              >
                <Form.Item
                  name="confirmPassword"
                  rules={[
                    {
                      required: true,
                      message: "Please confirm your password!",
                    },
                  ]}
                >
                  <Input.Password
                    placeholder="Confirm Password"
                    className={styles["password-input"]}
                  />
                </Form.Item>
              </div>
              <div className={styles["button-container"]}>
                <Button
                  type="primary"
                  htmlType="submit"
                  className={styles["submit-button"]}
                >
                  Izveidot kontu
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
          <div className={styles["log-in-container"]}>
            <span className={styles["log-in-text"]}>
              Vai tev jau ir konts? <a href="#">Autorizēties</a>
            </span>
          </div>
        </div>
      </Modal>
    </>
  );
};

export default SignUpModal;
