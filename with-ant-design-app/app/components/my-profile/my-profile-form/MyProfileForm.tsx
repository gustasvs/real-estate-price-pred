"use client";

import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { updateUserProfile as updateUserProfileApi } from "../../../../actions/user";

import imageCompression from "browser-image-compression";
import {
  Button,
  Col,
  Form,
  Input,
  message,
  Row,
  Upload,
  UploadProps,
} from "antd";

import styles from "./MyProfileForm.module.css";
import UserIcon from "../../user-icon/UserIcon";
import { EditOutlined, UploadOutlined } from "@ant-design/icons";
import { MDCTextField } from "@material/textfield";

import InputLabel from "../../input-fields/InputFields";
import { styled, TextField } from "@mui/material";

import AvatarEditor from 'react-avatar-editor'
import Dropzone from 'react-dropzone'
import { profile } from "console";



export const StyledTextField = styled(TextField)({

  '& label': {
    color: '#A0AAB4',
  },

  '& input': {
    color: '#6F7E8C',
  },

  '& label.Mui-focused': {
    color: '#A0AAB4',
  },
  '& .MuiInput-underline:after': {
    borderBottomColor: '#B2BAC2',
  },
  '& .MuiOutlinedInput-root': {
    borderRadius: '1em',
    '& fieldset': {
      borderColor: '#E0E3E7',
    },
    '&:hover fieldset': {
      borderColor: '#B2BAC2',
    },
    '&.Mui-focused fieldset': {
      borderColor: '#6F7E8C',
    },
  },
});


const props: UploadProps = {
  name: "file",
  multiple: true,
  listType: "picture",
  // action: 'https://www.mocky.io/v2/5cc8019d300000980a055e76',
  onChange(info) {
    const { status } = info.file;
    if (status !== "uploading") {
      console.log(info.file, info.fileList);
    }
    if (status === "done") {
      message.success(`${info.file.name} file uploaded successfully.`);
    } else if (status === "error") {
      message.error(`${info.file.name} file upload failed.`);
    }
  },
  onDrop(e) {
    console.log("Dropped files", e.dataTransfer.files);
  },
};

const MyProfileForm = () => {
  const { data: session, status, update } = useSession();
  const router = useRouter();

  const [form] = Form.useForm();

  const [fileList, setFileList] = useState([]);

  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  const handleUpdateProfile = async (values: any) => {
    console.log("Received values of form: ", values);

    const picturePromises = values?.pictures?.map(async (file: any) => {
      const compressedFile = await imageCompression(file.originFileObj, {
        maxSizeMB: 0.05,
        maxWidthOrHeight: 400,
      });

      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(compressedFile); // Convert compressed image to base64
        reader.onload = () => resolve(reader.result);
        reader.onerror = (error) => reject(error);
      });
    });
    const pictures = picturePromises ? await Promise.all(picturePromises) : [];

    const picture = pictures.find(p => p) || null;

    console.log("picture", picture);

    const res = await updateUserProfileApi({
      name: values.name,
      email: values.email,
      password: values.password || "",
      confirmPassword: values.confirmPassword || "",
      image: picture,
      id: session?.user?.id || "",
    });

    console.log("res", res);

    if ("error" in res) {
      message.error("Failed to update profile");
    } else {
      message.success("Profile updated successfully");
      await update();
    }
  };

  const normFile = (e: any) => {
    if (Array.isArray(e)) {
      return e;
    }
    return e && e.fileList;
  };

  const handleChange = ({ fileList: newFileList }) => setFileList(newFileList);

  useEffect(() => {
    // fill form with user data
    if (session) {
      console.log("filling form with user data");
      form.setFieldsValue({
        name: session.user?.name,
        email: session.user?.email,
      });
    
      setUserPicture(session.user?.image || "");
    }
  }, [session]);

  useEffect(() => {

    console.log("fieldValues", form.getFieldsValue());

  }, [form]);

  const [userPicture, setUserPicture] = useState<string | File>(session?.user?.image || "");

  const handleDrop = (dropped) => {
    console.log("dropped", dropped);
    setUserPicture(URL.createObjectURL(dropped[0]));
  }

  const [editorHovered, setEditorHovered] = useState(false);

  return (
    <Form
      initialValues={{
        name: session?.user?.name ? session?.user?.name : "",
        email: session?.user
          ? session?.user?.email
          : "",
      }
      }
      form={form}
      layout="horizontal"
      onFinish={handleUpdateProfile}
      className={styles["form-container"]}
    >
      <div className={styles["form-container"]}>
        <div className={styles["profile-summary-container"]}>
          <Form.Item
            name="pictures"
            //   label={<span className={styles["label"]}>Pictures</span>}
            valuePropName="fileList"
            getValueFromEvent={normFile}
          >
            <div
              className={styles["profile-image-container"]}
              onMouseEnter={() => setEditorHovered(true)}
              onMouseLeave={() => setEditorHovered(false)}
            >
              <div className={styles["profile-image-edit"]}>
                <EditOutlined 
                  onClick={() => document.getElementById('avatarUpload')?.click()}
                />
              <input
                type="file"
                id="avatarUpload"
                style={{ display: 'none' }}
                accept="image/*"
                onChange={(event) => {
                  if (!event.target.files) return;
                  const file = event.target.files[0];
                  if (file) {
                    setUserPicture(URL.createObjectURL(file));
                  }
                }}
              />
            </div>
            <AvatarEditor
              image={userPicture || ""}
              width={200}
              height={200}
              backgroundColor="transparent"
              border={editorHovered ? 50 : 0}
              // color={[255, 255, 255, 0.6]}
              // scale={1.2}
              rotate={0}
              borderRadius={0}
              className={`${styles["profile-image"]} ${editorHovered ? styles["profile-image-hovered"] : ""}`}
            />
            </div>
          </Form.Item>

          {/* </div> */}
          <div className={styles["profile-summary"]}>
            <div className={styles["profile-name"]}>
              <span>{session?.user?.name || "example name"}</span>
            </div>
            <div className={styles["profile-email"]}>
              <span>{session?.user?.email || "example email"}</span>
            </div>
          </div>
        </div>

        <div className={styles["text-field-container"]}>
          <Row gutter={8} justify="center">
            {/* First row: Name and Email */}
            {console.log(session)}
            <Col span={8}>
              <Form.Item
                name="name"
              >
                <StyledTextField
                  id="outlined-basic"
                  label="Vārds"
                  variant="outlined"
                  defaultValue={session?.user?.name}
                  placeholder={session?.user?.name || "Lietotāja vārds"}
                />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                name="email"
                rules={[{ required: true, message: "Please input the email!" }]}
              >
                <StyledTextField
                  id="outlined-basic"
                  label="E-pasts"
                  variant="outlined"
                  placeholder={session?.user?.email || "Lietotāja e-pasts"}
                  defaultValue={session?.user?.email}
                />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={12} justify="center">
            {/* Second row: Password and Confirm Password */}
            <Col span={8}>
              <Form.Item
                name="password"
              >
                <StyledTextField
                  id="outlined-basic"
                  label="Jauna parole"
                  variant="outlined"
                  // type="password"
                  onChange={(e) =>
                    setShowConfirmPassword(e.target.value.length > 0)
                  }
                />
              </Form.Item>
            </Col>
            <Col span={8}>
              {showConfirmPassword && (
                <Form.Item
                  name="confirmPassword"
                  rules={[
                    { required: true, message: "Please confirm the password!" },
                    ({ getFieldValue }) => ({
                      validator(_, value) {
                        if (!value || getFieldValue("password") === value) {
                          return Promise.resolve();
                        }
                        return Promise.reject(
                          new Error("Passwords do not match!")
                        );
                      },
                    }),
                  ]}
                >
                  <StyledTextField
                    id="outlined-basic"
                    label="Apstiprināt jauno paroli"
                    variant="outlined"
                    type="password"
                  />
                </Form.Item>
              )}
            </Col>
          </Row>
        </div>

        <Button
          type="primary"
          htmlType="submit"
          style={{ marginTop: "1.5rem" }}
          className={styles["submit-button"]}
        >
          Saglabāt izmaiņas
        </Button>
      </div>
    </Form>
  );
};

export default MyProfileForm;
