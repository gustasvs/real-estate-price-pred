"use client";

import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { updateUserProfile as updateUserProfileApi } from "../../../../actions/user";

import imageCompression from "browser-image-compression";
import {
  Button,
  Col,
  Divider,
  Form,
  Input,
  message,
  Popover,
  Row,
  Upload,
  UploadProps,
} from "antd";

import styles from "./MyProfileForm.module.css";
import UserIcon from "../../user-icon/UserIcon";
import { EditOutlined, QuestionCircleOutlined, UploadOutlined } from "@ant-design/icons";
import { MDCTextField } from "@material/textfield";

import InputLabel from "../../input-fields/InputFields";
import { Slider, styled, TextField } from "@mui/material";

import AvatarEditor from 'react-avatar-editor'
import Dropzone from 'react-dropzone'
import { profile } from "console";
import { saveImageOnCloud } from "../../../../actions/cloud_storage_helpers";
import { generateUploadUrl } from "../../../api/generateUploadUrl";
import { generateDownloadUrl } from "../../../api/generateDownloadUrl";



export const StyledTextField = styled(TextField)({

  '& .MuiTextField-root': {
    // margin: '1em 0',
    width: '100%',
  },

  '& .MuiInputBase-root': {
    backgroundColor: "var(--background-dark-main-hover)",
  },

  '& label': {
    color: "var(--background-light-main)",
  },

  '& input': {
    color: "white",
  },

  '& label.Mui-focused': {
    color: "var(--background-light-main)",
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
    // Error state
    '&.Mui-error fieldset': {
      borderColor: 'red', // Red border for error state
    },
    '&.Mui-error:hover fieldset': {
      borderColor: 'darkred', // Darker red border on hover in error state
    },
    '&.Mui-error.Mui-focused fieldset': {
      borderColor: 'darkred', // Darker red border when focused in error state
    },
  },
  // palceholder color
  '& .MuiOutlinedInput-input': {
    color: "var(--text-brighter)",
  },

  '&.Mui-error fieldset': {
    borderColor: 'red', // This line adds the red border when there is an error
  },

  // helper text color
  '& .MuiFormHelperText-root': {
    textAlign: 'right',
    color: "var(--background-light-main)",
  },

});

export const StyledNumberInput = styled(TextField)({

  '& .MuiInputBase-root': {
    backgroundColor: "var(--background-dark-main-hover)",
  },

  '& label': {
    color: "var(--background-light-main)",
  },

  '& input': {
    color: "var(--background-light-secondary)",
    MozAppearance: 'textfield', // Removes arrows in Firefox
  },

  '& input[type=number]': {
    WebkitAppearance: 'none', // Removes arrows in Chrome
    margin: 0, // Ensures consistency in alignment
  },

  '& label.Mui-focused': {
    color: "var(--background-light-main)",
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

import Switch from '@mui/material/Switch';
import { MdWidthFull } from "react-icons/md";
import { text } from "stream/consumers";

export const StyledSwitch = styled(Switch)(({ theme }) => ({
  '& .MuiSwitch-switchBase': {
    color: "var(--background-light-secondary)",
    '&:hover': {
      backgroundColor: 'rgba(0, 0, 0, 0.1)',
    },
    '&.Mui-checked': {
      color: "var(--background-light-main)",
      '&:hover': {
        backgroundColor: 'rgba(0, 0, 0, 0.2)',
      },
    },
  },
  '& .MuiSwitch-track': {
    backgroundColor: '#E0E3E7',
    opacity: 1,
    borderRadius: 16,
  },
  '& .Mui-checked + .MuiSwitch-track': {
    backgroundColor: '#6F7E8C',
    opacity: 1,
  },
}));



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

  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  const [userPicture, setUserPicture] = useState<string | File>("");

  const [imageScale, setImageScale] = useState(1);

  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  const handleUpdateProfile = async (values: any) => {
    console.log("Received values of form: ", values);

    console.log("uploadedFile", uploadedFile);

    let uploadedImageUrl = null;

    if (uploadedFile) {
      try {
        // Step 1: Get Pre-signed PUT URL
        const uploadUrlResults = await generateUploadUrl(uploadedFile.name, "profile-pictures");

        if (typeof uploadUrlResults !== "object" || "error" in uploadUrlResults) {
          message.error("Failed to get upload URL");
          return;
        }

        const { presignedUrl, objectKey } = uploadUrlResults;

        if (typeof presignedUrl !== "string" || !objectKey) {
          message.error("Failed to get upload URL");
          return;
        }

        console.log("Pre-signed Upload URL:", presignedUrl);

        // Step 2: Upload Image Using Pre-signed URL
        const formData = new FormData();
        formData.append("file", uploadedFile);

        const uploadResponse = await fetch(presignedUrl, {
          method: "PUT",
          body: uploadedFile, // Send the raw file
        });

        if (!uploadResponse.ok) {
          message.error("Failed to upload image to MinIO");
          return;
        }

        // Save the unique object key (or MinIO file URL) for further processing
        uploadedImageUrl = `${objectKey}`; // Optionally, prepend the MinIO URL if required

        console.log("Uploaded Image URL:", uploadedImageUrl);

      } catch (error) {
        console.error("Error uploading image to MinIO:", error);
        message.error("Image upload failed");
        return;
      }
    }

    // Step 3: Save Profile Information with Uploaded Image URL
    const res = await updateUserProfileApi({
      name: values.name,
      email: values.email,
      password: values.password || "",
      confirmPassword: values.confirmPassword || "",
      image: uploadedImageUrl,
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

  useEffect(() => {
    // fill form with user data
    if (session) {
      console.log("filling form with user data");
      console.log("session", session?.user);
      form.setFieldsValue({
        name: session.user?.name,
        email: session.user?.email,
      });
    
      setUserPicture(session.user?.image || "");
    }
  }, [session]);


  useEffect(() => {
    if (session?.user?.image) {
      const fetchUserImage = async () => {
        const sessionUserImage = session?.user?.image;
        if (sessionUserImage) {
          const downloadUrl = await generateDownloadUrl(sessionUserImage, "profile-pictures");

        console.log("Download URL:", downloadUrl);
        if (typeof downloadUrl === "object" && "error" in downloadUrl) {
          } else {
            setUserPicture(downloadUrl);
          }
        }
      };

      fetchUserImage();
    }
  }
  , [session]);

  const [editorHovered, setEditorHovered] = useState(false);


  console.log("status", status);
  if (status === "loading") {
    return <div>Loading...</div>;
  }

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
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "1rem",
        height: "100%",
        width: "100%",
        paddingBottom: "1rem",
      }}
      
    >
      <div style={{
        display: "flex",
        flexDirection: "row",
        gap: "1rem",
        justifyContent: "center",
        height: "100%",
      }}>
      <div className={styles["form-container"]}>
        <div className={styles["profile-summary-container"]}>
          <Form.Item
            name="pictures"
            valuePropName="fileList"
            getValueFromEvent={normFile}
          >
            <div
              className={styles["profile-image-container"]}
            >
              {editorHovered && (
                  <div className={styles["profile-image-edit-buttons"]} >
                    <Slider
                      value={imageScale}
                      onChange={(event, value) => {
                        if (typeof value !== "number") return;
                        setImageScale(value);
                        console.log("value", value)
                      }}
                      min={1}
                      max={2}
                      step={0.01}
                      style={{ width: "100%" }}
                      sx={(t) => ({
                        color: 'var(--background-dark-main)',
                        height: 4,
                        '& .MuiSlider-thumb': {
                          width: 8,
                          height: 8,
                          color: 'var(--text-bright)',
                          transition: '0.3s cubic-bezier(.47,1.64,.41,.8)',
                          '&::before': {
                            boxShadow: '0 2px 12px 0 var(--background-light-secondary)',
                          },
                          '&:hover, &.Mui-focusVisible': {
                            boxShadow: `0px 0px 0px 8px ${'rgb(var(--background-light-secondary) / 16%)'}`,
                          },
                          '&.Mui-active': {
                            width: 20,
                            height: 20,
                          },
                        },
                        '& .MuiSlider-rail': {
                          opacity: 0.28,
                        },
                        '& .MuiSlider-track': {
                          // height: 4,
                          color: 'var(--background-light-main)',
                        },
                      })}
                    />
                    <div className={styles["profile-image-edit-buttons-buttons"]}>
                    <Button 
                      onClick={() => {
                        setEditorHovered(false)
                        setUserPicture(session?.user?.image || "")
                      }}
                    >
                      Atcelt
                    </Button>
                    <Button 
                      type="primary"
                      onClick={() => setEditorHovered(false)}
                    >
                      Apstiprināt
                    </Button>
                    <Popover
                      placement="top"
                      content={(
                        <span className={styles["profile-image-edit-buttons-info-text"]}>
                          Šī poga tikai apstiprina jaunās bildes novietojumu. Saglabāt var nospiežot "Saglabāt izmaiņas" pogu.
                        </span>
                      )} 

                    >
                      <div className={styles["profile-image-edit-buttons-info"]}>
                    <QuestionCircleOutlined />
                    </div>
                    </Popover>
                    </div>
                  </div> 
                )}
              <div className={`${styles["profile-image-edit"]} ${styles[`${editorHovered ? "hidden" : ""}`]}`}>
                <EditOutlined 
                  onClick={() => {
                    setEditorHovered(true)
                    document.getElementById('avatarUpload')?.click()
                  }}
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
                    // use url so user can immediately see the selected image
                    setUserPicture(URL.createObjectURL(file)); 
                    console.log("setting picture", file);
                    // form.setFieldsValue({ pictures: [file] }); // this unfortunately doesn't work due to antd limitations
                    setUploadedFile(file); // so we use a state instead
                    
                  }
                }}
              />
            </div>
            <AvatarEditor
              image={userPicture || "https://static.vecteezy.com/system/resources/previews/030/504/836/non_2x/avatar-account-flat-isolated-on-transparent-background-for-graphic-and-web-design-default-social-media-profile-photo-symbol-profile-and-people-silhouette-user-icon-vector.jpg"}
              width={210}
              height={210}
              scale={imageScale}
              border={editorHovered ? 70 : 0}
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
          <div style={{
            display: "flex",
            flexWrap: "wrap",
          }}>
            <div
              style={{
                display: "flex",
                flex: "50%",
              }}
            >
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
            </div>
            <div
              style={{
                display: "flex",
                flex: "50%",
              }}
            >
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
            </div>
          </div>

          <div style={{
            display: "flex",
            flexWrap: "wrap",
          }}>
            <div
              style={{
                display: "flex",
                flex: "50%",
              }}
            >
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
            </div>
            <div
              style={{
                display: "flex",
                flex: "50%",
              }}
            >
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
            </div>
          </div>
        </div>
      </div>

      <Divider type="vertical" style={{ borderColor: "var(--background-light-main)", height: "100%" }} />
          
      <div
        style={{
          width: "30%",
          paddingLeft: "2rem",
        }}
      >
        <div className={styles["section-title"]}>Personalizācija</div>
        <Form.Item
          name="font-size"
        >
          <StyledTextField
            id="outlined-basic"
            label="Fonta izmērs"
            variant="outlined"
            // type="password"
            onChange={(e) =>
              setShowConfirmPassword(e.target.value.length > 0)
            }
            style={{ width: "100%" }}
          />
        </Form.Item>
        <Form.Item
          name="font-size"
        >
          <StyledTextField
            id="outlined-basic"
            label="Fonta izmērs"
            variant="outlined"
            // type="password"
            onChange={(e) =>
              setShowConfirmPassword(e.target.value.length > 0)
            }
            style={{ width: "100%" }}
          />
        </Form.Item>
      </div>

      </div>

      <div
        className={styles["main-button-container"]}
      >

      <Button
        style={{ marginTop: "1.5rem" }}
        className={`${styles["revert-button"]} ${styles["main-button"]}`}
      >
        Atcelt izmaiņas
      </Button>

      <Button
          type="primary"
          htmlType="submit"
          style={{ marginTop: "1.5rem" }}
          className={`${styles["submit-button"]} ${styles["main-button"]}`}
        >
          Saglabāt izmaiņas
        </Button>
      </div>
    </Form>
  );
};

export default MyProfileForm;
