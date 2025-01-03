"use client";

import React, { useEffect, useState } from "react";
import { useRouter, useParams } from "next/navigation";
import {
  Button,
  Form,
  Input,
  Upload,
  message,
  UploadProps,
  InputNumber,
  Row,
  Col,
  Tooltip,
} from "antd";
import {
  InboxOutlined,
  QuestionCircleOutlined,
  UploadOutlined,
} from "@ant-design/icons";
import { BsSignNoParking } from "react-icons/bs";

import imageCompression from "browser-image-compression";

import styles from "./ResidenceObjectForm.module.css";
import {
  createObject as createObjectApi,
  getObject as getObjectApi,
  updateObject as updateObjectApi,
} from "../../../actions/groupObjects";
import { generateUploadUrl } from "../../api/generateUploadUrl";
import { useSession } from "next-auth/react";
import { StyledNumberInput, StyledTextField } from "../styled-mui-components/styled-components";
import { ElevatorOutlined, NotAccessible } from "@mui/icons-material";
import FavouriteButton from "./FavouriteButton";
import { useThemeContext } from "../../context/ThemeContext";

const { Dragger } = Upload;

const ResidenceObjectForm = ({
  objectId,
  groupId,
  residence,
}: {
  objectId: string;
  groupId: string;
  residence: any;
}) => {
  const router = useRouter();

  const { data: session, status, update } = useSession();

  const { theme } = useThemeContext();

  const [form] = Form.useForm();

  const [parkingAvailable, setParkingAvailable] = useState<boolean>(residence?.parkingAvailable || false);
  const [elevatorAvailable, setElevatorAvailable] = useState<boolean>(residence?.elevatorAvailable || false);
  const [favourite, setFavourite] = useState<boolean>(residence?.favourite || false);

  const props: UploadProps = {
    name: "file",
    multiple: true,
    listType: "picture",
    maxCount: 10,
    onChange(info) {
      const { status } = info.file;
      if (status !== "uploading") {
        console.log(info.file, info.fileList);
      }
      if (status === "done") {
        message.success(
          `${info.file.name} fails augšupielādēts veiksmīgi!`
        );
      } else if (status === "error") {
        message.error(
          `${info.file.name} file upload failed.`
        );
      }
    },
    onDrop(e) {
      console.log("Dropped files", e.dataTransfer.files);
    },
    beforeUpload: (file) => {
      console.log("Before Upload", file);
      // Prevent files from auto-uploading
      return false;
    },
  };

  useEffect(() => {
    if (residence) {
      form.setFieldsValue(residence);
    }
  }, [residence]);
  

  const handleSubmit = async (values: any) => {
    console.log("Received values of form: ", values);

    const processedValues = {
      ...values,
      roomCount: Number(values.roomCount),
      parkingAvailable: Boolean(values.parkingAvailable),
      floor: Number(values.floor),
      buildingFloors: Number(values.buildingFloors),
      elevatorAvailable: Boolean(values.elevatorAvailable),
      favourite: Boolean(values.favourite),
      area: Number(values.area),
      price: Number(values.price),
    };

    const submitFunction =
      objectId !== "new"
        ? updateObjectApi
        : createObjectApi;

    const picturePromises = values?.pictures?.map(
      async (file: any) => {
        console.log("FILE", file);
        // if the file has an originFileObj, it means it's a new file
        if (file.originFileObj) {
          const compressedFile = await imageCompression(
            file.originFileObj,
            {
              maxSizeMB: 0.05,
              maxWidthOrHeight: 400,
            }
          );

          try {
            // Step 1: Get Pre-signed PUT URL
            const uploadUrlResults =
              await generateUploadUrl(
                compressedFile.name,
                "object-pictures"
              );

            if (
              typeof uploadUrlResults !== "object" ||
              "error" in uploadUrlResults
            ) {
              throw new Error("Failed to get upload URL");
            }

            const { presignedUrl, objectKey } =
              uploadUrlResults;

            if (
              typeof presignedUrl !== "string" ||
              !objectKey
            ) {
              throw new Error("Failed to get upload URL");
            }

            console.log(
              "Pre-signed Upload URL:",
              presignedUrl
            );

            // Step 2: Upload Image Using Pre-signed URL
            const uploadResponse = await fetch(
              presignedUrl,
              {
                method: "PUT",
                body: compressedFile, // Send the raw file
              }
            );

            if (!uploadResponse.ok) {
              throw new Error(
                "Failed to upload image to MinIO"
              );
            }

            console.log("Uploaded Image Key:", objectKey);

            // Step 3: Return the object key as the picture URL
            return { pictureUrl: objectKey, status: "new" };
          } catch (error) {
            console.error(
              "Error uploading image to MinIO:",
              error
            );
            message.error("Image upload failed");
            throw error; // Propagate error to halt processing
          }
        } else {
          return {
            pictureUrl: file.picture,
            status: "unchanged",
          };
        }
      }
    );

    // Step 4: Now we push the new objects values and keys of the images to our postgreSQL database
    try {
      const pictures = await Promise.all(picturePromises);
      console.log("Uploaded Pictures:", pictures);

      // create needs groupId, update needs objectId
      const idToUse = objectId === "new" ? groupId : objectId;

      const res = await submitFunction(idToUse as string, {
        ...processedValues,
        pictures,
      });

      console.log("res", res);

      if ("error" in res) {
        message.error("Failed to create object");
      } else {
        message.success(
          objectId !== "new"
            ? "Objekts atjaunots veiksmīgi!"
            : "Objekts izveidots veiksmīgi!"
        );
        router.push(`/groups/${groupId}`);
      }
    } catch (error) {
      console.error("Error processing images:", error);
    }
  };

  const normFile = (e: any) => {
    if (Array.isArray(e)) {
      return e;
    }
    return e && e.fileList;
  };

  if (status === "loading") {
    return <div>Loading...</div>;
  }

  return (
    <Form
      form={form}
      layout="horizontal"
      onFinish={handleSubmit}
      scrollToFirstError
      className={styles["form-container"]}
    >
      <Row gutter={[84, 64]}>
        {/* Left Side */}
        <Col span={12}>
          {/* Name */}
          <Form.Item
            name="name"
            // label={<span className={styles["label"]}>Name</span>}
            rules={[
              {
                required: true,
                message:
                  "Lūdzu ievadiet kā dēvēsiet šo īpašumu!",
              },
            ]}
            className={styles["form-item"]}
          >
            {/* <Input placeholder="Enter the name" className={styles["input-field"]} /> */}
            <StyledTextField
              style={{ width: "100%" }}
              id="filled-basic-name"
              label="Objekta nosaukums"
              variant="outlined"
              defaultValue={residence?.name}
              placeholder="Ievadiet objekta nosaukumu"
            />
          </Form.Item>

          {/* Description */}
          <Form.Item
            name="description"
            // label={<span className={styles["label"]}>Description</span>}
            rules={[
              {
                required: true,
                message:
                  "Lūdzu ievadiet aprakstu!",
              },
            ]}
          >
            <StyledTextField
              style={{ width: "100%" }}
              id="outlined-basic"
              label="Apraksts"
              variant="outlined"
              defaultValue={residence?.description}
              placeholder="Ievadiet objekta aprakstu"
              multiline
              rows={3}
              // error={form.getFieldError('description').length > 0}
              // error={true}
              // error={!!form.getFieldError("description").length} // Checks if there are errors
              // helperText={form.getFieldError("description")}
              // helperText="Aprakstam jābūt vismaz 10 simbolus garumā"
            />
          </Form.Item>

          {/* Address */}
          <Form.Item
            name="address"
            // label={<span className={styles["label"]}>Address</span>}
            rules={[
              {
                required: true,
                message: "Lūdzu ievadiet objekta adresi!",
              },
            ]}
          >
            <StyledTextField
              style={{ width: "100%" }}
              id="outlined-basic"
              label="Adrese"
              variant="outlined"
              defaultValue={residence?.address}
              placeholder="Ievadiet objekta adresi"
            />
          </Form.Item>

          {/* Area */}
          <Form.Item
            name="area"
            rules={[
              {
                required: true,
                message: "Lūdzu ievadiet objekta platību!",
              },
            ]}
          >
            <StyledNumberInput
              style={{ width: "100%" }}
              id="outlined-basic"
              label="Platība (kvadrātmetri)"
              variant="outlined"
              defaultValue={residence?.area}
              placeholder="Ievadiet objekta platību"
            />
          </Form.Item>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="roomCount"
                rules={[
                  {
                    required: true,
                    message: "Lūdzu ievadiet istabu skaitu!",
                    transform: (value) => (value ? Number(value) : value),
                  },
                ]}
              >
                <StyledNumberInput
                  style={{ width: "100%" }}
                  id="outlined-basic"
                  label="Istabu skaits"
                  variant="outlined"
                  defaultValue={residence?.roomCount}
                  // type="number"
                  placeholder="Ievadiet istabu skaitu"
                />
              </Form.Item>
            </Col>

            {/* Parking Count */}
            <Col span={12}>
              <Form.Item
                name="parkingAvailable"
              >
                <div
                  className={styles["parking-container"]}
                  onClick={() => {
                    setParkingAvailable((prev: boolean) => {
                      form.setFieldsValue({ parkingAvailable: !prev });
                      return prev ? false : true;
                    });
                  }}
                >
                  <Tooltip title={parkingAvailable ? "Stāvvietas ir pieejamas" : "Stāvvietas nav pieejamas"}>
                    <QuestionCircleOutlined
                      style={{
                        backgroundColor: "var(--background-dark-main)",
                        position: "absolute",
                        top: -10,
                        right: -10,
                        color: "var(--background-light-main)",
                        fontSize: "1.5em",
                        cursor: "pointer",
                        borderRadius: "40%",
                      }}
                    />
                  </Tooltip>
                  <label
                    style={{
                      color: "var(--background-light-main)",
                      // fontWeight: "bold",
                      paddingLeft: "1em",
                      fontSize: ".9rem",
                      cursor: "pointer",
                      height: "1.4375em",
                      padding: "16.5px 14px 16.5px 14px",
                    }}
                  >
                    Stāvvietas pieejamiba
                  </label>
                  {parkingAvailable ? (
                  <div
                    className={`${styles[`parking-icon`]} ${styles["parking-available"]}
                      ${styles[`${theme !== "dark" ? "parking-available-light" : ""}`]}`}
                  ></div>
                  ) : (
                    <BsSignNoParking 
                    // className={`${styles["parking-icon"]} ${styles["parking-unavailable"]}`}
                    className={styles["parking-icon"]}
                    style={{
                      // red tint
                      height: "2.5rem",
                      fill: "#ff0000",
                      background: "var(--background-dark-main-hover)",
                      // filter: "invert(19%) sepia(95%) saturate(7481%) hue-rotate(360deg) brightness(96%) contrast(110%)",
                    }}/>
                  )}
                </div>
              </Form.Item>
            </Col>

            <Col span={8}>
              <Form.Item
                name="floor"
                dependencies={["buildingFloors"]}
                rules={[
                  {
                    required: true,
                    message: "Lūdzu ievadiet stāvu!",
                  },
                  ({ getFieldValue }) => ({
                    validator(_, value) {
                      if (!value || getFieldValue('buildingFloors') >= value) {
                        return Promise.resolve();
                      }
                      return Promise.reject(new Error('Stāvam jābūt mazākam par ēkas stāvu skaitu!'));
                    },
                  }),
                ]}
                
              >
                <StyledNumberInput
                  style={{ width: "100%" }}
                  id="outlined-basic"
                  label="Stāvs"
                  variant="outlined"
                  defaultValue={residence?.floor}
                  placeholder="Ievadiet stāvu"
                />
              </Form.Item>
            </Col>

            <Col span={8}>
              <Form.Item
                name="buildingFloors"
                rules={[
                  {
                    required: true,
                    message: "Lūdzu ievadiet ēkas stāvu skaitu!",
                  },
                ]}
              >
                <StyledNumberInput
                  style={{ width: "100%" }}
                  id="outlined-basic"
                  label="Ēkas stāvu skaits"
                  variant="outlined"
                  defaultValue={residence?.buildingFloors}
                  placeholder="Ievadiet ēkas stāvu skaitu"
                />
              </Form.Item>
            </Col>

            <Col span={8}>
              <Form.Item
                name="elevatorAvailable"
              >
                <div
                  className={styles["parking-container"]}
                  onClick={() => {
                    setElevatorAvailable((prev: boolean) => {
                      form.setFieldsValue({ elevatorAvailable: !prev });
                      return prev ? false : true;
                    });
                  }}
                >
                  <Tooltip title={elevatorAvailable ? "Lifts ir pieejams" : "Lifts nav pieejams"}>
                    <QuestionCircleOutlined
                      style={{
                        backgroundColor: "var(--background-dark-main)",
                        position: "absolute",
                        top: -10,
                        right: -10,
                        color: "var(--background-light-main)",
                        fontSize: "1.5em",
                        cursor: "pointer",
                        borderRadius: "40%",
                      }}
                    />
                  </Tooltip>
                  <label
                    style={{
                      color: "var(--background-light-main)",
                      // fontWeight: "bold",
                      paddingLeft: "1em",
                      fontSize: ".9rem",
                      cursor: "pointer",
                      height: "1.4375em",
                      padding: "11.5px 14px 21.5px 14px",
                      lineHeight: "1",
                      textAlign: "center",
                    }}
                  >
                    Lifta pieejamība
                  </label>
                  {elevatorAvailable ? 
                    <ElevatorOutlined style={{ color: "yellowgreen", fontSize: "2.5rem", marginRight: "1rem", background: "var(--background-dark-main-hover)" }} 
                    className={styles["parking-icon"]} /> 
                    : 
                    <NotAccessible style={{ fill: "red", marginRight: "1rem", width: "2.5rem", height: "2.5rem", background: "var(--background-dark-main-hover)" }} className={styles["parking-icon"]}/>
                  }
                </div>
              </Form.Item>
            </Col>

          </Row>
          {/* Price */}
          <Form.Item
            name="price"
            rules={[
              {
                required: true,
                message: "Lūdzu ievadiet objekta cenu!",
              },
            ]}
          >
            <StyledNumberInput
              style={{ width: "100%" }}
              id="outlined-basic"
              label="Tirgus cena (EUR)"
              variant="outlined"
              defaultValue={residence?.price}
              placeholder="Ievadiet objekta cenu"
            />
          </Form.Item>
        {/*  */}
        {objectId !== "new" && (
          <Form.Item name="favourite">
            <Tooltip title={"Atzīmē šo dzīvokli, un vēlāk to varēsi ātri atrast favorītu sarakstā!"}>
                    <QuestionCircleOutlined
                      style={{
                        backgroundColor: "var(--background-dark-main)",
                        position: "absolute",
                        top: -10,
                        right: -10,
                        color: "var(--background-light-main)",
                        fontSize: "1.5em",
                        cursor: "pointer",
                        borderRadius: "40%",
                      }}
                    />
                  </Tooltip>
            <FavouriteButton
              style={{ 
                backgroundColor: "var(--background-dark-main-hover",
                border: "1px solid var(--background-light-main)",
                borderRadius: "1em",
                height: "1.4375em",
                padding: "16.5px 14px 16.5px 14px",
              }}
              fromForm={true}
              favourite={favourite}
              onClick={(e) => {
                setFavourite((prev: boolean) => {
                  form.setFieldsValue({ favourite: !prev });
                  return prev ? false : true;
                });
              }}
            />
          </Form.Item>
        )}

        </Col>
        {/* Right Side */}
        <Col span={12}>
          {/* Pictures */}
          <Form.Item
            name="pictures"
            // label={<span className={styles["label"]}>Pictures</span>}
            valuePropName="fileList"
            getValueFromEvent={normFile}
            rules={[
              {
                required: true,
                message: "Lūdzu augšupielādējiet bildes!",
              },
            ]}
          >
            {/* <MemoizedDragger {...props} /> */}
            <Dragger {...props}>
              <p className={styles["upload-drag-icon"]}>
                <InboxOutlined />
              </p>
              <p className={styles["upload-text"]}>
                {/* Click or drag file to this area to upload */}
                Spied peles klikšķi vai arī ievelc šeit bildes, lai tās augšupielādētu
              </p>
              <p className={styles["upload-hint"]}>
                {/* Support for a single or bulk upload.
                Strictly prohibited from uploading company
                data or other banned files. */}
                Var augšupielādēt gan vienu, gan vairākas bildes.
              {/* </p>
              <p className={styles["upload-hint"]}> */}
               {" "}Publiskotās bildes tiek automātiski uzskatītas kā publisks īpašums. Uzmanies, lai tajās nebūtu tava personīgā informācija!
              </p>
              <div className={styles["upload-hint"]}>
                {/* {form.getFieldValue("pictures")?.length || 0}/10 bildes augšupielādētas */}
                Var augšupielādēt līdz 10 bildēm*
              </div>
            </Dragger>

          </Form.Item>
        </Col>
      </Row>
      {/* Buttons */}
      <div className={styles["button-container"]}>
        <Button
          type="default"
          onClick={() => router.push(`/groups/${groupId}`)}
          className={styles["cancel-button"]}
        >
          {objectId !== "new" ? "Atgriezties uz grupu" : "Atcelt izveidi un atgriezties uz grupu"}
        </Button>

        <Button
          type="primary"
          htmlType="submit"
          className={styles["submit-button"]}
        >
          {objectId === "new"
            ? "Izveidot jaunu objektu"
            : "Saglabāt izmaiņas"}
        </Button>
      </div>
    </Form>
  );
};

export default ResidenceObjectForm;
