FROM containerregistry-na.jpmchase.net/container-base/managedbaseimages/python:3.10.7-stable

RUN microdnf install -y python3-devel gcc gcc-c++ make zlib-devel openssl-devel krb5-libs krb5-devel cyrus-sasl-gssapi cyrus-sasl-devel wget unzip yum bzip2-devel libffi-devel xz-devel

COPY requirements.txt ./requirements.txt

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


RUN mkdir -p /app/logs

WORKDIR /home/jpmcnobody/nltk_data/
COPY resources/nltk/nltk_data/ .

# Establish working directory
WORKDIR /app/
# Copy application to container's app folder
COPY app .

# Change ownership to 'jpmcnobody' user aka account with least permissions
RUN chown jpmcnobody /app/
RUN chmod 777 -R /app/logs
RUN chown jpmcnobody /home/jpmcnobody/nltk_data/
RUN chmod 777 -R /home/jpmcnobody/nltk_data

# Run this container as 'jpmcnobody' user.
USER 999
ENV PYTHONPATH /

#check Dirs
RUN ls -la /app
RUN ls -la /app/model
RUN ls /home/jpmcnobody/nltk_data


# Command to run application
CMD [ "python", "/app/wsgi.py" ]

