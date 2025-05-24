# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import urllib.request


def download_file(url, filename=None):
    """Downloads a file from a URL using urllib.request.

    Args:
        url: The URL of the file to download.
        filename: The optional name to save the file as. If None, it's extracted from the URL.

    Returns:
        The path to the downloaded file.
    """
    if filename is None:
        filename = os.path.basename(url)
    try:
        if not url.startswith(("https:", "http:")):
            raise ValueError("URL must start with 'http:' or 'https:'")
        urllib.request.urlretrieve(url, filename)  # noqa S310
        return filename
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None
