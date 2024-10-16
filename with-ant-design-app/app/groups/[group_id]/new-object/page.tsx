"use client";


import React from 'react';
import { useRouter } from 'next/router';
import { Button, Form, Input, Upload, message } from 'antd';
import { UploadOutlined } from '@ant-design/icons';
import GenericLayout from '../../../components/generic-page-layout';

const NewObjectForm = () => {
  const router = useRouter();
  const { group_id } = router.query;
  const [form] = Form.useForm();

  const handleSubmit = async (values: any) => {
    console.log('Received values of form: ', values);
    // API call to create the object
  };

  const normFile = (e: any) => {
    if (Array.isArray(e)) {
      return e;
    }
    return e && e.fileList;
  };

  return (
    <GenericLayout>
      <Form
        form={form}
        layout="vertical"
        onFinish={handleSubmit}
        style={{ maxWidth: 600, margin: 'auto' }}
      >
        <Form.Item
          name="name"
          label="Name"
          rules={[{ required: true, message: 'Please input the name of the real estate object!' }]}
        >
          <Input placeholder="Enter the name" />
        </Form.Item>
        <Form.Item
          name="description"
          label="Description"
          rules={[{ required: true, message: 'Please input the description!' }]}
        >
          <Input.TextArea rows={4} placeholder="Enter the description" />
        </Form.Item>
        <Form.Item
          name="pictures"
          label="Pictures"
          valuePropName="fileList"
          getValueFromEvent={normFile}
          extra="Upload pictures of the real estate object"
        >
          <Upload
            name="logo"
            action="/upload.do"
            listType="picture"
            beforeUpload={() => false} // Prevent automatic uploading
          >
            <Button icon={<UploadOutlined />}>Click to upload</Button>
          </Upload>
        </Form.Item>
        <Form.Item>
          <Button type="primary" htmlType="submit">
            Submit
          </Button>
        </Form.Item>
      </Form>
    </GenericLayout>
  );
};

export default NewObjectForm;
