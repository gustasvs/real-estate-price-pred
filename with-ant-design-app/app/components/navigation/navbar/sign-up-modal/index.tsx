import React, { useEffect, useState } from "react";
import { Alert, Button, Divider, Form, Input, message, Modal } from "antd";
import {
  GithubOutlined,
  GoogleOutlined,
  TwitterCircleFilled,
} from "@ant-design/icons";
import styles from "./SignUpModal.module.css";
import { signIn, signOut, useSession } from "next-auth/react";
import { useSearchParams } from "next/navigation";
import { useRouter } from "next/navigation";

interface SignUpModalProps {
  open: boolean;
  setOpen: (open: boolean) => void;
  setLoginModalOpen: (open: boolean) => void;
}

const SignUpModal: React.FC<SignUpModalProps> = ({ open, setOpen, setLoginModalOpen }) => {

  const { data: session } = useSession() as { data: CustomSession | null };
  const searchParams = useSearchParams();

  const router = useRouter();

  const [error, setError] = useState<string | null>(null);

  const [infoMessage, setInfoMessage] = useState<string | null>(null);

  interface CustomSession {
      error?: string;
      message?: string;
    }
  
  

  const [form] = Form.useForm();

  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  const [passwordsMatch, setPasswordsMatch] = useState(false);

  const handleCloseModal = () => {
    setOpen(false);
    setError(null);
    setInfoMessage(null);
    form.resetFields();

    const params = new URLSearchParams(searchParams);
    params.delete("modal");
    router.push(`?${params.toString()}`, { scroll: false });
  };

  const passwordStrength = (password: string) => {
    if (password.length < 8) {
      return 0;
    }
    return 4;
  };

  const handlePasswordChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (passwordStrength(event.target.value)) {
      setShowConfirmPassword(true);
    } else {
      setShowConfirmPassword(false);
    }
  };

  const handleConfirmPasswordChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.value === event.target.value) {
      setPasswordsMatch(true);
    } else {
      setPasswordsMatch(false
      );
    }
  };


  const handleSignUp = async (values: {
    email: string;
    password: string;
    confirmPassword: string;
  }) => {
    if (values.password !== values.confirmPassword) {
      setError("Ievadītās paroles nesakrīt!");
      return;
    }
    
    const result = await signIn("credentials", {
      email: values.email,
      password: values.password,
      confirmPassword: values.confirmPassword,
      redirect: false, // Prevent automatic navigation on error
    });

    console.log("result", result);

    // if (result && "error" in result) {
    //   setError("Kļūda reģistrējoties! Lūdzu pārbaudiet ievadītos datus, vai mēģiniet vēlreiz vēlāk.");
      // setError(result.error);
    // } else {
    //   setError(null);
      // router.push("/profile");
  };
  
  
  useEffect(() => {
    if (session && session.error) {
      setError(session.error as string);
    }
    else if (session && session.message) {
      setError(null);
      setInfoMessage(session.message);
      message.success("Reģistrācija veiksmīga!");
    } else {
      setError(null);
    }
  }, [session]);


  useEffect(() => {
    if (searchParams.get("modal") === "sign-up") {
      setOpen(true);
    }
  }, [searchParams]);

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
              {/* Izveido kontu izmantojot sociālos tīklus */}
              Izveido kontu izmantojot e-pastu un paroli
            </span>
            {/* <div className={styles["soc-login-buttons"]}>
              <GoogleOutlined
                className={`${styles["soc-login-icon"]} ${styles["google-icon"]}`}
                onClick={() => {
                  // signIn("google", { callbackUrl: "/" });
                }}
              />

              <TwitterCircleFilled className={styles["soc-login-icon"]} />

              <GithubOutlined className={styles["soc-login-icon"]} />
            </div> */}
          </div>
          {/* <div className={styles["divider-container"]}>
            <Divider plain>vai</Divider>
          </div> */}
          <div className={styles["email-sign-up-container"]}>
            <Form
              name="basic"
              initialValues={{ remember: true }}
              onFinish={handleSignUp}
              style={{ width: "100%" }}
              form={form}
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
                      message: "Lūdzu ievadiet atkārtotu paroli!",
                    },
                  ]}
                >
                  <Input.Password
                    placeholder="Atkārtota parole"
                    className={styles["password-input"]}
                  />
                </Form.Item>
              </div>

                {error && (
                  <Alert
                    message={error}
                    type="error"
                    showIcon
                    style={{ marginBottom: "1em" }}
                  />
                )}
                {infoMessage && (
                  <Alert
                    message={infoMessage}
                    type="info"
                    showIcon
                    style={{ marginBottom: "1em" }}
                  />
                )}

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
              Vai tev jau ir konts? <a onClick={() => {handleCloseModal(); setLoginModalOpen(true)} } className={styles["footer-link"]}>Autorizēties</a>
            </span>
          </div>
        </div>
      </Modal>
    </>
  );
};

export default SignUpModal;
